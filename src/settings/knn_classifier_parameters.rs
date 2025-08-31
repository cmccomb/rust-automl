//! KNN classifier parameters

use crate::utils::distance::Distance;
pub use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};

/// Parameters for k-nearest neighbors (KNN) classification
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KNNClassifierParameters {
    /// Number of nearest neighbors to use
    pub(crate) k: usize,
    /// Weighting function to use with KNN regression
    pub(crate) weight: KNNWeightFunction,
    /// Search algorithm to use with KNN regression
    pub(crate) algorithm: KNNAlgorithmName,
    /// Distance metric to use with KNN regression
    pub(crate) distance: Distance,
}

impl KNNClassifierParameters {
    /// Define the number of nearest neighbors to use
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Define the weighting function to use with KNN regression
    #[must_use]
    pub const fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }

    /// Define the search algorithm to use with KNN regression
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Define the distance metric to use with KNN regression
    #[must_use]
    pub const fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }
}

impl Default for KNNClassifierParameters {
    fn default() -> Self {
        Self {
            k: 3,
            weight: KNNWeightFunction::Uniform,
            algorithm: KNNAlgorithmName::CoverTree,
            distance: Distance::Euclidean,
        }
    }
}
