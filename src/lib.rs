//! Rust core of the x3d xarray backend. Python bindings are thin: all I/O, shard-index handling,
//! decompression and hyperslab assembly happen here with the GIL released.

mod array;
mod error;
mod store;

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::array::{Array, Meta, Slice};
use crate::store::{ByteRange, Store};

fn slices(sel: Vec<(u64, u64, u64)>) -> Vec<Slice> {
    sel.into_iter().map(|(start, stop, step)| Slice { start, stop, step }).collect()
}

/// A store root (HTTP(S) URL, `file://` URL or directory).
#[pyclass(name = "Store", frozen)]
struct PyStore(Store);

#[pymethods]
impl PyStore {
    #[new]
    fn new(root: &str) -> Self {
        PyStore(Store::open(root))
    }

    fn child(&self, key: &str) -> PyStore {
        PyStore(self.0.child(key.trim_matches('/')))
    }

    #[getter]
    fn location(&self) -> String {
        self.0.location()
    }

    /// Whole object, or None if it does not exist.
    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let r = py.detach(|| self.0.get(key, ByteRange::Full))?;
        Ok(r.map(|b| PyBytes::new(py, &b)))
    }
}

/// One Zarr v3 array inside a store, built from its `zarr.json` metadata.
#[pyclass(name = "Array", frozen)]
struct PyArray(Array);

#[pymethods]
impl PyArray {
    #[new]
    fn new(store: &PyStore, path: &str, metadata: &str) -> PyResult<Self> {
        Ok(PyArray(Array::new(store.0.clone(), path, Meta::parse(metadata)?)))
    }

    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.0.meta.shape.clone()
    }

    #[getter]
    fn chunks(&self) -> Vec<u64> {
        self.0.meta.chunks.clone()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.0.meta.dtype.clone()
    }

    /// Estimated (requests, uncompressed bytes) for a read of `sel` = [(start, stop, step), ...].
    fn plan(&self, sel: Vec<(u64, u64, u64)>) -> PyResult<(u64, u64)> {
        let p = self.0.plan(&slices(sel))?;
        Ok((p.requests, p.bytes))
    }

    /// Flat uint8 array holding the C-ordered selection; view it as `dtype` and reshape.
    fn read_raw<'py>(&self, py: Python<'py>, sel: Vec<(u64, u64, u64)>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let sel = slices(sel);
        let out = py.detach(|| self.0.read(&sel))?;
        Ok(out.into_pyarray(py))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    m.add_class::<PyArray>()?;
    Ok(())
}
