//! Error types for distance calculations.

use std::fmt::{Display, Formatter};

use super::Distance;

/// Errors that can occur when working with distance metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceError {
    /// Distance metric is not supported in this context.
    UnsupportedDistance(Distance),
}

impl Display for DistanceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDistance(d) => write!(f, "unsupported distance: {d}"),
        }
    }
}

impl std::error::Error for DistanceError {}
