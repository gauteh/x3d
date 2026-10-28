//! Zarr v3 array reader: regular chunk grid, `bytes` (little endian) + optional `zstd`,
//! and `sharding_indexed` (with `crc32c`-protected index). Reads strided hyperslabs.

use crate::error::{Error, Result};
use crate::store::{ByteRange, Store};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Concurrency is bound by network latency, not CPU, so use more threads than cores
/// (`X3D_THREADS` overrides).
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("X3D_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(16));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("x3d-io-{i}"))
            .build()
            .expect("thread pool")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    pub start: u64,
    pub stop: u64,
    pub step: u64,
}

impl Slice {
    pub fn len(&self) -> u64 {
        if self.stop <= self.start { 0 } else { (self.stop - self.start).div_ceil(self.step) }
    }
}

/// Chunk indices (along one dimension) containing at least one selected element.
fn touched(s: &Slice, chunk: u64) -> Vec<u64> {
    if s.len() == 0 {
        return vec![];
    }
    let last = s.start + (s.len() - 1) * s.step;
    (s.start / chunk..=last / chunk)
        .filter(|&c| {
            // first selected element at or after the chunk start must be inside the chunk
            let lo = c * chunk;
            let k = if lo <= s.start { 0 } else { (lo - s.start).div_ceil(s.step) };
            s.start + k * s.step < lo + chunk && k < s.len()
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Codecs {
    zstd: bool,
}

#[derive(Debug, Clone)]
enum Layout {
    Plain,
    Sharded { inner: Vec<u64>, index_at_start: bool },
}

#[derive(Debug, Clone)]
pub struct Meta {
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
    pub itemsize: usize,
    fill: Vec<u8>,
    codecs: Codecs,
    layout: Layout,
}

fn bad(m: impl Into<String>) -> Error {
    Error::Metadata(m.into())
}

fn u64_vec(v: Option<&Value>, what: &str) -> Result<Vec<u64>> {
    v.and_then(Value::as_array)
        .ok_or_else(|| bad(format!("missing {what}")))?
        .iter()
        .map(|x| x.as_u64().ok_or_else(|| bad(format!("bad {what}"))))
        .collect()
}

fn dtype_size(dt: &str) -> Result<usize> {
    Ok(match dt {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" => 8,
        _ => return Err(Error::Unsupported(format!("data_type {dt}"))),
    })
}

fn fill_bytes(dt: &str, v: &Value) -> Result<Vec<u8>> {
    let float = |f: f64| -> Vec<u8> {
        if dt == "float32" { (f as f32).to_le_bytes().to_vec() } else { f.to_le_bytes().to_vec() }
    };
    let n = dtype_size(dt)?;
    if dt.starts_with("float") {
        let f = match v {
            Value::Number(x) => x.as_f64().unwrap_or(0.0),
            Value::String(s) => match s.as_str() {
                "NaN" => f64::NAN,
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => return Err(Error::Unsupported(format!("fill_value {s}"))),
            },
            _ => 0.0,
        };
        return Ok(float(f));
    }
    let i: i128 = match v {
        Value::Number(x) => x.as_i64().map(i128::from).or(x.as_u64().map(i128::from)).unwrap_or(0),
        Value::Bool(b) => *b as i128,
        _ => 0,
    };
    Ok(i.to_le_bytes()[..n].to_vec())
}

impl Meta {
    pub fn parse(json: &str) -> Result<Meta> {
        let v: Value = serde_json::from_str(json).map_err(|e| bad(e.to_string()))?;
        if v["zarr_format"] != 3 || v["node_type"] != "array" {
            return Err(Error::Unsupported("only zarr v3 arrays".into()));
        }
        let shape = u64_vec(v.get("shape"), "shape")?;
        let dtype = v["data_type"].as_str().ok_or_else(|| bad("data_type"))?.to_string();
        let itemsize = dtype_size(&dtype)?;
        let grid = &v["chunk_grid"];
        if grid["name"] != "regular" {
            return Err(Error::Unsupported("chunk_grid must be regular".into()));
        }
        let chunks = u64_vec(grid["configuration"].get("chunk_shape"), "chunk_shape")?;
        let ke = &v["chunk_key_encoding"];
        if ke["name"] != "default" || ke["configuration"]["separator"].as_str().unwrap_or("/") != "/" {
            return Err(Error::Unsupported("chunk_key_encoding must be default with '/'".into()));
        }
        if chunks.len() != shape.len() || chunks.contains(&0) {
            return Err(bad("chunk_shape does not match shape"));
        }
        let fill = fill_bytes(&dtype, &v["fill_value"])?;

        let codecs = v["codecs"].as_array().ok_or_else(|| bad("codecs"))?;
        let (layout, codecs) = match codecs.first() {
            Some(c) if c["name"] == "sharding_indexed" => {
                let cfg = &c["configuration"];
                let inner = u64_vec(cfg.get("chunk_shape"), "inner chunk_shape")?;
                if inner.len() != chunks.len() || inner.contains(&0) || chunks.iter().zip(&inner).any(|(c, i)| c % i != 0) {
                    return Err(bad("inner chunk_shape must evenly divide the shard shape"));
                }
                let names: Vec<_> = cfg["index_codecs"]
                    .as_array()
                    .map(|a| a.iter().map(|c| c["name"].as_str().unwrap_or("")).collect())
                    .unwrap_or_default();
                if names != ["bytes", "crc32c"] {
                    return Err(Error::Unsupported(format!("index_codecs {names:?}")));
                }
                let at_start = match cfg["index_location"].as_str().unwrap_or("end") {
                    "start" => true,
                    "end" => false,
                    l => return Err(Error::Unsupported(format!("index_location {l}"))),
                };
                let inner_codecs = parse_codecs(cfg["codecs"].as_array().ok_or_else(|| bad("inner codecs"))?)?;
                (Layout::Sharded { inner, index_at_start: at_start }, inner_codecs)
            }
            _ => (Layout::Plain, parse_codecs(codecs)?),
        };
        Ok(Meta { shape, chunks, dtype, itemsize, fill, codecs, layout })
    }

    /// Shape of the unit that is fetched with one request (inner chunk if sharded).
    fn leaf_shape(&self) -> &[u64] {
        match &self.layout {
            Layout::Plain => &self.chunks,
            Layout::Sharded { inner, .. } => inner,
        }
    }
}

fn parse_codecs(list: &[Value]) -> Result<Codecs> {
    let names: Vec<_> = list.iter().map(|c| c["name"].as_str().unwrap_or("")).collect();
    match names.as_slice() {
        ["bytes"] | ["bytes", "zstd"] => {}
        _ => return Err(Error::Unsupported(format!("codecs {names:?}"))),
    }
    let endian = list[0]["configuration"]["endian"].as_str().unwrap_or("little");
    if endian != "little" {
        return Err(Error::Unsupported("big endian".into()));
    }
    Ok(Codecs { zstd: names.len() == 2 })
}

type Index = Option<Arc<Vec<(u64, u64)>>>;

/// Where one read spent its time. The `*_ns` fields are summed over all threads, so compared with
/// `wall_ns` they show how much work was done in parallel (and what the work consisted of).
#[derive(Debug, Default)]
pub struct Profile {
    /// Shard index requests (cold ones only).
    pub index_ns: AtomicU64,
    /// Waiting for the store: HTTP round trip + body, or the file read.
    pub fetch_ns: AtomicU64,
    pub decode_ns: AtomicU64,
    /// Copying a decoded chunk into the output, including the wait for the output lock.
    pub scatter_ns: AtomicU64,
    pub chunks: AtomicU64,
    pub bytes_fetched: AtomicU64,
    pub wall_ns: AtomicU64,
}

fn since(t: Instant, into: &AtomicU64) {
    into.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
}

/// CF unpacking done while the chunks are decoded: `int16` -> `float32` as `raw * scale`, `fill` -> NaN.
/// It is the same arithmetic as xarray's `mask_and_scale` (a float32 scale), done once, per chunk, in
/// parallel and straight into the output, instead of three passes over the whole result in numpy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decode {
    pub scale: f32,
    pub fill: Option<i16>,
}

impl Decode {
    fn value(&self, v: i16) -> f32 {
        if Some(v) == self.fill { f32::NAN } else { v as f32 * self.scale }
    }

    fn apply(&self, raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(raw.len() * 2);
        for c in raw.chunks_exact(2) {
            out.extend_from_slice(&self.value(i16::from_le_bytes([c[0], c[1]])).to_le_bytes());
        }
        out
    }
}

pub struct Array {
    store: Store,
    path: String,
    pub meta: Meta,
    decode: Option<Decode>,
    indexes: Mutex<HashMap<String, Index>>,
}

/// Cost of a read: requests issued (shard indexes counted as cold) and uncompressed bytes of the
/// chunks touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub requests: u64,
    pub bytes: u64,
}

/// Shard index entry of a chunk that was never written.
const ABSENT: (u64, u64) = (u64::MAX, u64::MAX);

struct Leaf {
    /// Element offset of the leaf chunk in the array.
    origin: Vec<u64>,
    key: String,
    /// Location inside the object: None = whole object.
    span: Option<(u64, u64)>,
}

impl Array {
    pub fn new(store: Store, path: &str, meta: Meta) -> Array {
        Array { store, path: path.trim_matches('/').to_string(), meta, decode: None, indexes: Mutex::new(HashMap::new()) }
    }

    /// Return `float32` (unpacked) instead of the stored `int16`.
    pub fn with_decode(mut self, decode: Decode) -> Result<Array> {
        if self.meta.dtype != "int16" {
            return Err(Error::Unsupported(format!("unpacking {}", self.meta.dtype)));
        }
        self.decode = Some(decode);
        Ok(self)
    }

    /// Bytes per element of what `read` returns.
    pub fn out_itemsize(&self) -> usize {
        if self.decode.is_some() { 4 } else { self.meta.itemsize }
    }

    pub fn check(&self, sel: &[Slice]) -> Result<()> {
        if sel.len() != self.meta.shape.len() {
            return Err(Error::Selection(format!("need {} slices, got {}", self.meta.shape.len(), sel.len())));
        }
        for (s, &n) in sel.iter().zip(&self.meta.shape) {
            if s.step == 0 || s.stop > n || s.start > s.stop {
                return Err(Error::Selection(format!("slice {}:{}:{} out of range for {n}", s.start, s.stop, s.step)));
            }
        }
        Ok(())
    }

    pub fn plan(&self, sel: &[Slice]) -> Result<Plan> {
        self.check(sel)?;
        let count = |chunk: &[u64]| -> u64 { sel.iter().zip(chunk).map(|(s, &c)| touched(s, c).len() as u64).product() };
        let leaves = count(self.meta.leaf_shape());
        let requests = match self.meta.layout {
            Layout::Plain => leaves,
            Layout::Sharded { .. } => count(&self.meta.chunks) + leaves,
        };
        let elems: u64 = self.meta.leaf_shape().iter().product();
        Ok(Plan { requests, bytes: leaves * elems * self.meta.itemsize as u64 })
    }

    fn key(&self, idx: &[u64]) -> String {
        let mut k = format!("{}/c", self.path);
        for i in idx {
            k.push('/');
            k.push_str(&i.to_string());
        }
        k
    }

    fn shard_index(&self, key: &str) -> Result<Index> {
        if let Some(i) = self.indexes.lock().unwrap().get(key) {
            return Ok(i.clone());
        }
        let Layout::Sharded { inner, index_at_start } = &self.meta.layout else { unreachable!() };
        let n: u64 = self.meta.chunks.iter().zip(inner).map(|(c, i)| c / i).product();
        let size = n * 16 + 4;
        let range = if *index_at_start { ByteRange::Span { start: 0, len: size } } else { ByteRange::Suffix(size) };
        let idx = match self.store.get(key, range)? {
            None => None,
            Some(b) => {
                if b.len() as u64 != size {
                    return Err(Error::Corrupt(format!("{key}: short shard index")));
                }
                let (body, crc) = b.split_at(b.len() - 4);
                if crc32c::crc32c(body) != u32::from_le_bytes(crc.try_into().unwrap()) {
                    return Err(Error::Corrupt(format!("{key}: shard index checksum mismatch")));
                }
                Some(Arc::new(
                    body.chunks_exact(16)
                        .map(|e| (u64::from_le_bytes(e[..8].try_into().unwrap()), u64::from_le_bytes(e[8..].try_into().unwrap())))
                        .collect::<Vec<_>>(),
                ))
            }
        };
        self.indexes.lock().unwrap().insert(key.to_string(), idx.clone());
        Ok(idx)
    }

    /// Leaves (fetchable chunks) of one outer chunk that the selection touches.
    fn leaves(&self, sel: &[Slice], outer: &[u64], prof: &Profile) -> Result<Vec<Leaf>> {
        let key = self.key(outer);
        let Layout::Sharded { inner, .. } = &self.meta.layout else {
            return Ok(vec![Leaf { origin: outer.iter().zip(&self.meta.chunks).map(|(i, c)| i * c).collect(), key, span: None }]);
        };
        let t = Instant::now();
        let index = self.shard_index(&key)?;
        since(t, &prof.index_ns);
        let grid: Vec<u64> = self.meta.chunks.iter().zip(inner).map(|(c, i)| c / i).collect();
        // a missing shard is all absent chunks
        let index = index.unwrap_or_else(|| vec![ABSENT; grid.iter().product::<u64>() as usize].into());
        let base: Vec<u64> = outer.iter().zip(&self.meta.chunks).map(|(i, c)| i * c).collect();
        // Per dimension, inner chunk indices (within the shard) that are selected.
        let per_dim: Vec<Vec<u64>> = (0..sel.len())
            .map(|d| {
                touched(&sel[d], inner[d])
                    .into_iter()
                    .filter(|&g| g * inner[d] >= base[d] && g * inner[d] < base[d] + self.meta.chunks[d])
                    .map(|g| g - base[d] / inner[d])
                    .collect()
            })
            .collect();
        let mut out = vec![];
        for_each_index(&per_dim, |ix| {
            let lin = ix.iter().zip(&grid).fold(0u64, |acc, (&i, &g)| acc * g + i);
            let (off, len) = index[lin as usize];
            out.push(Leaf {
                origin: (0..ix.len()).map(|d| base[d] + ix[d] * inner[d]).collect(),
                key: key.clone(),
                span: Some((off, len)),
            });
        });
        Ok(out)
    }

    fn fetch(&self, leaf: &Leaf, prof: &Profile) -> Result<Option<Vec<u8>>> {
        if leaf.span == Some(ABSENT) {
            return Ok(None);
        }
        let range = match leaf.span {
            None => ByteRange::Full,
            Some((start, len)) => ByteRange::Span { start, len },
        };
        let t = Instant::now();
        let raw = self.store.get(&leaf.key, range)?;
        since(t, &prof.fetch_ns);
        let Some(raw) = raw else { return Ok(None) };
        prof.bytes_fetched.fetch_add(raw.len() as u64, Relaxed);
        prof.chunks.fetch_add(1, Relaxed);
        let t = Instant::now();
        let expect = self.meta.leaf_shape().iter().product::<u64>() as usize * self.meta.itemsize;
        let data = if self.meta.codecs.zstd {
            zstd::bulk::decompress(&raw, expect).map_err(|e| Error::Corrupt(format!("{}: {e}", leaf.key)))?
        } else {
            raw
        };
        if data.len() != expect {
            return Err(Error::Corrupt(format!("{}: chunk is {} bytes, expected {expect}", leaf.key, data.len())));
        }
        let data = match &self.decode {
            Some(d) => d.apply(&data),
            None => data,
        };
        since(t, &prof.decode_ns);
        Ok(Some(data))
    }

    /// Read a strided hyperslab as C-ordered little-endian bytes. Missing chunks read as the fill value.
    pub fn read(&self, sel: &[Slice]) -> Result<Vec<u8>> {
        self.read_profiled(sel).map(|(out, _)| out)
    }

    pub fn read_profiled(&self, sel: &[Slice]) -> Result<(Vec<u8>, Profile)> {
        use rayon::prelude::*;
        let started = Instant::now();
        let prof = Profile::default();
        self.check(sel)?;
        let out_shape: Vec<u64> = sel.iter().map(Slice::len).collect();
        let n: usize = out_shape.iter().product::<u64>() as usize;
        let isz = self.out_itemsize();
        let mut out = vec![0u8; n * isz];
        let fill = match &self.decode {
            Some(d) => d.value(i16::from_le_bytes([self.meta.fill[0], self.meta.fill[1]])).to_le_bytes().to_vec(),
            None => self.meta.fill.clone(),
        };
        if n == 0 {
            return Ok((out, prof));
        }

        let per_dim: Vec<Vec<u64>> = sel.iter().zip(&self.meta.chunks).map(|(s, &c)| touched(s, c)).collect();
        let mut outer = vec![];
        for_each_index(&per_dim, |ix| outer.push(ix.to_vec()));

        let out = Mutex::new(out);
        let leaf_shape = self.meta.leaf_shape().to_vec();
        pool().install(|| {
            outer.par_iter().try_for_each(|o| -> Result<()> {
                let leaves = self.leaves(sel, o, &prof)?;
                leaves.par_iter().try_for_each(|leaf| -> Result<()> {
                    // every selected element lies in a leaf, so the output is written once (no pre-fill);
                    // absent chunks are scattered as fill
                    let data = match self.fetch(leaf, &prof)? {
                        Some(d) => d,
                        None if fill.iter().all(|&b| b == 0) => return Ok(()),
                        None => fill.repeat(leaf_shape.iter().product::<u64>() as usize),
                    };
                    let t = Instant::now();
                    let mut out = out.lock().unwrap();
                    scatter(&data, &leaf.origin, &leaf_shape, sel, &out_shape, isz, &mut out);
                    since(t, &prof.scatter_ns);
                    Ok(())
                })
            })
        })?;
        since(started, &prof.wall_ns);
        Ok((out.into_inner().unwrap(), prof))
    }
}

/// Visit the cartesian product of per-dimension index lists (C order).
fn for_each_index(per_dim: &[Vec<u64>], mut f: impl FnMut(&[u64])) {
    if per_dim.iter().any(Vec::is_empty) {
        return;
    }
    let mut pos = vec![0usize; per_dim.len()];
    let mut cur: Vec<u64> = per_dim.iter().map(|d| d[0]).collect();
    loop {
        f(&cur);
        let mut d = per_dim.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            pos[d] += 1;
            if pos[d] < per_dim[d].len() {
                cur[d] = per_dim[d][pos[d]];
                break;
            }
            pos[d] = 0;
            cur[d] = per_dim[d][0];
        }
    }
}

/// Copy the selected elements of one chunk (at `origin`, shape `cshape`) into the output.
fn scatter(chunk: &[u8], origin: &[u64], cshape: &[u64], sel: &[Slice], out_shape: &[u64], isz: usize, out: &mut [u8]) {
    let nd = sel.len();
    // Per dim: (first output index, count, first chunk-local index).
    let mut dims = Vec::with_capacity(nd);
    for d in 0..nd {
        let s = &sel[d];
        let lo = origin[d].max(s.start);
        let hi = (origin[d] + cshape[d]).min(s.stop);
        if hi <= lo {
            return;
        }
        let k0 = (lo - s.start).div_ceil(s.step);
        let first = s.start + k0 * s.step;
        if first >= hi {
            return;
        }
        let cnt = (hi - 1 - first) / s.step + 1;
        dims.push((k0 as usize, cnt as usize, (first - origin[d]) as usize));
    }
    let cstride = strides(cshape);
    let ostride = strides(out_shape);

    fn rec(d: usize, co: usize, oo: usize, dims: &[(usize, usize, usize)], sel: &[Slice], cs: &[usize], os: &[usize], isz: usize, chunk: &[u8], out: &mut [u8]) {
        let (k0, cnt, c0) = dims[d];
        let step = sel[d].step as usize;
        if d + 1 == dims.len() {
            if step == 1 {
                let (s, o) = ((co + c0) * isz, (oo + k0) * isz);
                out[o..o + cnt * isz].copy_from_slice(&chunk[s..s + cnt * isz]);
            } else {
                for k in 0..cnt {
                    let (s, o) = ((co + c0 + k * step) * isz, (oo + k0 + k) * isz);
                    out[o..o + isz].copy_from_slice(&chunk[s..s + isz]);
                }
            }
            return;
        }
        for k in 0..cnt {
            rec(d + 1, co + (c0 + k * step) * cs[d], oo + (k0 + k) * os[d], dims, sel, cs, os, isz, chunk, out);
        }
    }
    // The last dimension's stride is 1 element, so offsets are in elements.
    rec(0, 0, 0, &dims, sel, &cstride, &ostride, isz, chunk, out);
}

fn strides(shape: &[u64]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1] as usize;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sl(start: u64, stop: u64, step: u64) -> Slice {
        Slice { start, stop, step }
    }

    #[test]
    fn touched_skips_chunks_without_selected_elements() {
        assert_eq!(touched(&sl(0, 10, 1), 4), vec![0, 1, 2]);
        // elements 0, 9: chunks 0 and 2 only
        assert_eq!(touched(&sl(0, 10, 9), 4), vec![0, 2]);
        assert_eq!(touched(&sl(5, 5, 1), 4), Vec::<u64>::new());
        assert_eq!(touched(&sl(7, 8, 1), 4), vec![1]);
    }

    /// Hand-encoded 4x6 int16 array: one shard (index at start) of 2x3 inner chunks, zstd; one inner chunk absent.
    fn build(dir: &std::path::Path, at_start: bool) -> Array {
        let data: Vec<i16> = (0..24).collect();
        let (mut body, mut index) = (vec![], vec![]);
        let hdr = 4 * 16 + 4; // 2x2 inner chunks
        for (n, (r, c)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            if n == 2 {
                index.extend([u64::MAX.to_le_bytes(), u64::MAX.to_le_bytes()].concat());
                continue;
            }
            let mut raw = vec![];
            for i in 0..2 {
                for j in 0..3 {
                    raw.extend(data[(r * 2 + i) * 6 + c * 3 + j].to_le_bytes());
                }
            }
            let z = zstd::bulk::compress(&raw, 1).unwrap();
            let off = if at_start { hdr + body.len() } else { body.len() };
            index.extend((off as u64).to_le_bytes());
            index.extend((z.len() as u64).to_le_bytes());
            body.extend(z);
        }
        index.extend(crc32c::crc32c(&index).to_le_bytes());
        let obj = if at_start { [index, body].concat() } else { [body, index].concat() };
        std::fs::create_dir_all(dir.join("a/c/0")).unwrap();
        std::fs::write(dir.join("a/c/0/0"), obj).unwrap();
        let meta = format!(
            r#"{{"zarr_format":3,"node_type":"array","shape":[4,6],"data_type":"int16","fill_value":-1,
            "chunk_grid":{{"name":"regular","configuration":{{"chunk_shape":[4,6]}}}},
            "chunk_key_encoding":{{"name":"default","configuration":{{"separator":"/"}}}},
            "codecs":[{{"name":"sharding_indexed","configuration":{{"chunk_shape":[2,3],
              "codecs":[{{"name":"bytes","configuration":{{"endian":"little"}}}},{{"name":"zstd","configuration":{{"level":1}}}}],
              "index_codecs":[{{"name":"bytes","configuration":{{"endian":"little"}}}},{{"name":"crc32c"}}],
              "index_location":"{}"}}}}]}}"#,
            if at_start { "start" } else { "end" }
        );
        Array::new(Store::open(dir.to_str().unwrap()), "a", Meta::parse(&meta).unwrap())
    }

    fn as_i16(b: &[u8]) -> Vec<i16> {
        b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn sharded_read_both_index_locations() {
        for at_start in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let a = build(dir.path(), at_start);
            let all = as_i16(&a.read(&[sl(0, 4, 1), sl(0, 6, 1)]).unwrap());
            let want: Vec<i16> = (0..24).map(|v| if (2..4).contains(&(v / 6)) && v % 6 < 3 { -1 } else { v }).collect();
            assert_eq!(all, want);
            // strided across chunk borders
            let s = as_i16(&a.read(&[sl(1, 4, 2), sl(1, 6, 2)]).unwrap());
            assert_eq!(s, vec![7, 9, 11, -1, 21, 23]);
            // a single point
            assert_eq!(as_i16(&a.read(&[sl(3, 4, 1), sl(4, 5, 1)]).unwrap()), vec![22]);
        }
    }

    #[test]
    fn decode_unpacks_and_masks_like_cf() {
        let dir = tempfile::tempdir().unwrap();
        let raw = build(dir.path(), true);
        let a = Array::new(raw.store.clone(), "a", raw.meta.clone()).with_decode(Decode { scale: 0.5, fill: Some(7) }).unwrap();
        assert_eq!(a.out_itemsize(), 4);
        let out = a.read(&[sl(0, 4, 1), sl(0, 6, 1)]).unwrap();
        let got: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let stored = as_i16(&raw.read(&[sl(0, 4, 1), sl(0, 6, 1)]).unwrap());
        for (g, v) in got.iter().zip(stored) {
            // -1 is the array fill value (absent chunks); only 7 is the mask value
            if v == 7 { assert!(g.is_nan()) } else { assert_eq!(*g, v as f32 * 0.5) }
        }
        // absent chunks decode too: the array fill (-1) is not the mask value, so it becomes -0.5
        std::fs::remove_file(dir.path().join("a/c/0/0")).unwrap();
        let gone = Array::new(raw.store.clone(), "a", raw.meta.clone()).with_decode(Decode { scale: 0.5, fill: Some(7) }).unwrap();
        assert_eq!(gone.read(&[sl(0, 1, 1), sl(0, 1, 1)]).unwrap(), (-0.5f32).to_le_bytes());
    }

    #[test]
    fn decode_needs_int16() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = build(dir.path(), true);
        a.meta.dtype = "float32".into();
        assert!(matches!(a.with_decode(Decode { scale: 1.0, fill: None }), Err(Error::Unsupported(_))));
    }

    #[test]
    fn plan_counts_index_and_chunk_requests() {
        let dir = tempfile::tempdir().unwrap();
        let a = build(dir.path(), true);
        assert_eq!(a.plan(&[sl(0, 4, 1), sl(0, 6, 1)]).unwrap(), Plan { requests: 5, bytes: 4 * 6 * 2 });
        assert_eq!(a.plan(&[sl(0, 1, 1), sl(0, 1, 1)]).unwrap(), Plan { requests: 2, bytes: 6 * 2 });
    }

    #[test]
    fn missing_shard_is_fill() {
        let dir = tempfile::tempdir().unwrap();
        let a = build(dir.path(), true);
        std::fs::remove_file(dir.path().join("a/c/0/0")).unwrap();
        let a = Array::new(a.store.clone(), "a", a.meta.clone());
        assert_eq!(as_i16(&a.read(&[sl(0, 2, 1), sl(0, 2, 1)]).unwrap()), vec![-1; 4]);
    }

    #[test]
    fn corrupt_index_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let a = build(dir.path(), true);
        let p = dir.path().join("a/c/0/0");
        let mut b = std::fs::read(&p).unwrap();
        b[3] ^= 0xff;
        std::fs::write(&p, b).unwrap();
        let a = Array::new(a.store.clone(), "a", a.meta.clone());
        assert!(matches!(a.read(&[sl(0, 1, 1), sl(0, 1, 1)]), Err(Error::Corrupt(_))));
    }
}
