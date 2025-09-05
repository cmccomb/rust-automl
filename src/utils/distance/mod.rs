//! Distance metrics supported by the crate.

use std::convert::TryFrom;
use std::fmt::{Display, Formatter};

use smartcore::metrics::distance::{
    Distance as SmartcoreDistance, euclidian::Euclidian, hamming::Hamming, manhattan::Manhattan,
    minkowski::Minkowski,
};
use smartcore::numbers::basenum::Number;

/// Distance error types.
pub mod error;
pub use error::DistanceError;

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

/// Wrapper implementing [`SmartcoreDistance`] for KNN regressors.
#[derive(Clone, Debug)]
pub enum KNNRegressorDistance<T: Number> {
    /// Euclidean distance
    Euclidean(Euclidian<T>),
    /// Manhattan distance
    Manhattan(Manhattan<T>),
    /// Minkowski distance with parameter `p`
    Minkowski(Minkowski<T>),
    /// Hamming distance
    Hamming(Hamming<T>),
}

impl<T: Number> SmartcoreDistance<Vec<T>> for KNNRegressorDistance<T> {
    fn distance(&self, a: &Vec<T>, b: &Vec<T>) -> f64 {
        match self {
            Self::Euclidean(d) => d.distance(a, b),
            Self::Manhattan(d) => d.distance(a, b),
            Self::Minkowski(d) => d.distance(a, b),
            Self::Hamming(d) => d.distance(a, b),
        }
    }
}

impl<T: Number> TryFrom<Distance> for KNNRegressorDistance<T> {
    type Error = DistanceError;

    fn try_from(distance: Distance) -> Result<Self, Self::Error> {
        Ok(match distance {
            Distance::Euclidean => Self::Euclidean(Euclidian::new()),
            Distance::Manhattan => Self::Manhattan(Manhattan::new()),
            Distance::Minkowski(p) => Self::Minkowski(Minkowski::new(p)),
            Distance::Hamming => Self::Hamming(Hamming::new()),
            Distance::Mahalanobis => {
                return Err(DistanceError::UnsupportedDistance(distance));
            }
        })
    }
}

impl<T: Number> KNNRegressorDistance<T> {
    /// Create a distance wrapper from [`Distance`].
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::UnsupportedDistance`] if the distance is not supported.
    pub fn from(distance: Distance) -> Result<Self, DistanceError> {
        Self::try_from(distance)
    }
}
