use pyo3::PyErr;
use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("invalid metadata: {0}")]
    Metadata(String),
    #[error("corrupt data: {0}")]
    Corrupt(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("invalid selection: {0}")]
    Selection(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::NotFound(_) => PyFileNotFoundError::new_err(e.to_string()),
            Error::Io(_) => PyIOError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        }
    }
}
