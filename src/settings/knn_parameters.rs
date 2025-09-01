//! KNN parameters

use crate::utils::distance::Distance;
pub use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};

/// Parameters for k-nearest neighbor (KNN) algorithms
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KNNParameters {
    /// Number of nearest neighbors to use
    pub(crate) k: usize,
    /// Weighting function to use with KNN algorithms
    pub(crate) weight: KNNWeightFunction,
    /// Search algorithm to use with KNN algorithms
    pub(crate) algorithm: KNNAlgorithmName,
    /// Distance metric to use with KNN algorithms
    pub(crate) distance: Distance,
}

impl KNNParameters {
    /// Define the number of nearest neighbors to use
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Define the weighting function to use
    #[must_use]
    pub const fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }

    /// Define the search algorithm to use
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Define the distance metric to use
    #[must_use]
    pub const fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }
}

impl Default for KNNParameters {
    fn default() -> Self {
        Self {
            k: 3,
            weight: KNNWeightFunction::Uniform,
            algorithm: KNNAlgorithmName::CoverTree,
            distance: Distance::Euclidean,
        }
    }
}
