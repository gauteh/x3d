//! Byte-range access to a store root: HTTP(S) (public S3/RGW, nginx, ...) or a local directory.

use crate::error::{Error, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteRange {
    Full,
    /// `len` bytes from `start`.
    Span { start: u64, len: u64 },
    /// The last `len` bytes.
    Suffix(u64),
}

#[derive(Clone)]
pub enum Store {
    Http { base: String, agent: ureq::Agent },
    File { root: PathBuf },
}

const RETRIES: usize = 3;

impl Store {
    /// `root` is an `http(s)://` URL, a `file://` URL or a plain path.
    pub fn open(root: &str) -> Store {
        if root.starts_with("http://") || root.starts_with("https://") {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .http_status_as_error(false)
                .timeout_global(Some(Duration::from_secs(60)))
                .max_idle_connections_per_host(32)
                .build()
                .into();
            Store::Http { base: root.trim_end_matches('/').to_string(), agent }
        } else {
            let p = root.strip_prefix("file://").unwrap_or(root);
            Store::File { root: PathBuf::from(p) }
        }
    }

    /// A store rooted below this one (`key` has no leading/trailing slash).
    pub fn child(&self, key: &str) -> Store {
        match self {
            Store::Http { base, agent } => Store::Http { base: format!("{base}/{key}"), agent: agent.clone() },
            Store::File { root } => Store::File { root: root.join(key) },
        }
    }

    pub fn location(&self) -> String {
        match self {
            Store::Http { base, .. } => base.clone(),
            Store::File { root } => root.display().to_string(),
        }
    }

    /// `Ok(None)` if the object does not exist.
    pub fn get(&self, key: &str, range: ByteRange) -> Result<Option<Vec<u8>>> {
        match self {
            Store::File { root } => get_file(&root.join(key), range),
            Store::Http { base, agent } => {
                let url = format!("{base}/{key}");
                let mut last = String::new();
                for attempt in 0..RETRIES {
                    if attempt > 0 {
                        std::thread::sleep(Duration::from_millis(100 << attempt));
                    }
                    match get_http(agent, &url, range) {
                        Ok(r) => return Ok(r),
                        Err(Error::Io(e)) => last = e,
                        Err(e) => return Err(e),
                    }
                }
                Err(Error::Io(format!("{url}: {last}")))
            }
        }
    }
}

fn get_http(agent: &ureq::Agent, url: &str, range: ByteRange) -> Result<Option<Vec<u8>>> {
    let mut req = agent.get(url);
    match range {
        ByteRange::Full => {}
        ByteRange::Span { start, len } => {
            req = req.header("Range", format!("bytes={}-{}", start, start + len - 1));
        }
        ByteRange::Suffix(len) => req = req.header("Range", format!("bytes=-{len}")),
    }
    let resp = req.call().map_err(|e| Error::Io(e.to_string()))?;
    let status = resp.status().as_u16();
    match status {
        404 => Ok(None),
        200 | 206 => {
            let body = resp
                .into_body()
                .with_config()
                .limit(u64::MAX)
                .read_to_vec()
                .map_err(|e| Error::Io(e.to_string()))?;
            // A server ignoring Range answers 200 with the whole object.
            Ok(Some(match (status, range) {
                (200, ByteRange::Span { start, len }) => {
                    let s = (start as usize).min(body.len());
                    body[s..(s + len as usize).min(body.len())].to_vec()
                }
                (200, ByteRange::Suffix(len)) => body[body.len().saturating_sub(len as usize)..].to_vec(),
                _ => body,
            }))
        }
        s if s >= 500 => Err(Error::Io(format!("{url}: HTTP {s}"))),
        s => Err(Error::Metadata(format!("{url}: HTTP {s}"))),
    }
}

fn get_file(path: &std::path::Path, range: ByteRange) -> Result<Option<Vec<u8>>> {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::Io(format!("{}: {e}", path.display()))),
    };
    let io = |e: std::io::Error| Error::Io(format!("{}: {e}", path.display()));
    let mut buf = Vec::new();
    match range {
        ByteRange::Full => {
            f.read_to_end(&mut buf).map_err(io)?;
        }
        ByteRange::Span { start, len } => {
            f.seek(SeekFrom::Start(start)).map_err(io)?;
            f.take(len).read_to_end(&mut buf).map_err(io)?;
        }
        ByteRange::Suffix(len) => {
            let size = f.metadata().map_err(io)?.len();
            f.seek(SeekFrom::Start(size.saturating_sub(len))).map_err(io)?;
            f.read_to_end(&mut buf).map_err(io)?;
        }
    }
    Ok(Some(buf))
}
