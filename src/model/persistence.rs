//! Errors and filesystem helpers for persisted model artifacts.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io;
use std::path::{Path, PathBuf};

use atomic_write_file::AtomicWriteFile;
use serde::Serialize;
use serde_json::{Map, Value};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;

/// Errors produced while saving or loading a model artifact.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistenceError {
    /// No trained final model is available to save.
    NotTrained,
    /// A filesystem operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying I/O error message.
        message: String,
    },
    /// A trained model could not be encoded.
    Encode(String),
    /// A model artifact could not be decoded.
    Decode(String),
    /// The file is not a compatible `automl` model artifact.
    InvalidFormat(String),
    /// The artifact uses an unsupported format version.
    UnsupportedVersion {
        /// Newest format version understood by this crate version.
        supported: u32,
        /// Format version stored in the artifact.
        found: u32,
    },
    /// The selected trained algorithm does not expose persistable inference state.
    UnsupportedAlgorithm(String),
}

impl Display for PersistenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotTrained => write!(f, "no trained final model is available to save"),
            Self::Io {
                operation,
                path,
                message,
            } => write!(f, "could not {operation} at {}: {message}", path.display()),
            Self::Encode(message) => write!(f, "could not encode model artifact: {message}"),
            Self::Decode(message) => write!(f, "could not decode model artifact: {message}"),
            Self::InvalidFormat(message) => write!(f, "invalid model artifact: {message}"),
            Self::UnsupportedVersion { supported, found } => write!(
                f,
                "unsupported model artifact version {found}; this crate supports version {supported}"
            ),
            Self::UnsupportedAlgorithm(message) => {
                write!(f, "model persistence is not supported: {message}")
            }
        }
    }
}

impl Error for PersistenceError {}

/// Convenience result type for model persistence operations.
pub type PersistenceResult<T> = Result<T, PersistenceError>;

pub(crate) fn write_atomic<T>(path: &Path, value: &T) -> PersistenceResult<()>
where
    T: Serialize + ?Sized,
{
    let mut options = AtomicWriteFile::options();
    #[cfg(unix)]
    options.mode(0o600);

    let mut file = options
        .open(path)
        .map_err(|error| io_error("open temporary model artifact", path, &error))?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|error| {
        if error.is_io() {
            PersistenceError::Io {
                operation: "write temporary model artifact",
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        } else {
            PersistenceError::Encode(error.to_string())
        }
    })?;
    file.commit()
        .map_err(|error| io_error("commit model artifact", path, &error))
}

fn io_error(operation: &'static str, path: &Path, error: &io::Error) -> PersistenceError {
    PersistenceError::Io {
        operation,
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

pub(crate) fn required_field<'a>(
    value: &'a Value,
    field: &str,
    context: &str,
) -> Result<&'a Value, String> {
    as_object(value, context)?
        .get(field)
        .ok_or_else(|| format!("{context} is missing {field:?}"))
}

pub(crate) fn as_object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

pub(crate) fn as_array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], String> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{context} must be an array"))
}

pub(crate) fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{context} must be a finite number"))
}

pub(crate) fn usize_number(value: &Value, context: &str) -> Result<usize, String> {
    let number = value
        .as_u64()
        .ok_or_else(|| format!("{context} must be a non-negative integer"))?;
    usize::try_from(number).map_err(|_| format!("{context} is too large for this platform"))
}

pub(crate) fn numeric_array(value: &Value, context: &str) -> Result<usize, String> {
    let values = as_array(value, context)?;
    for (index, value) in values.iter().enumerate() {
        finite_number(value, &format!("{context}[{index}]"))?;
    }
    Ok(values.len())
}
