use crate::utils::Distance;
pub use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};

/// Parameters for k-nearest neighbors (KNN) classification
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KNNClassifierParameters {
    pub(crate) k: usize,
    pub(crate) weight: KNNWeightFunction,
    pub(crate) algorithm: KNNAlgorithmName,
    pub(crate) distance: Distance,
}

impl KNNClassifierParameters {
    /// Define the number of nearest neighbors to use
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Define the weighting function to use with KNN regression
    pub fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }

    /// Define the search algorithm to use with KNN regression
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Define the distance metric to use with KNN regression
    pub fn with_distance(mut self, distance: Distance) -> Self {
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
