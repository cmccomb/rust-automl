//! Shared model errors.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Errors that can occur when using models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// Attempted to use a model before training.
    NotTrained,
    /// The predicted cluster label could not be converted to the target type.
    InvalidClusterLabel(usize),
    /// Underlying algorithm failed during inference.
    Inference(String),
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotTrained => write!(f, "model has not been trained"),
            Self::InvalidClusterLabel(l) => write!(f, "invalid cluster label {l}"),
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
        }
    }
}

impl Error for ModelError {}

/// Convenience type alias for results produced by models.
pub type ModelResult<T> = Result<T, ModelError>;
