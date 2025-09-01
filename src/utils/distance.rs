//! Distance metrics supported by the crate.

use std::fmt::{Display, Formatter};

/// Distance metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Distance {
    /// Euclidean distance
    Euclidean,

    /// Manhattan distance
    Manhattan,

    /// Minkowski distance, parameterized by p
    Minkowski(u16),

    /// Mahalanobis distance
    Mahalanobis,

    /// Hamming distance
    Hamming,
}

impl Display for Distance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::Manhattan => write!(f, "Manhattan"),
            Self::Minkowski(n) => write!(f, "Minkowski(p = {n})"),
            Self::Mahalanobis => write!(f, "Mahalanobis"),
            Self::Hamming => write!(f, "Hamming"),
        }
    }
}
