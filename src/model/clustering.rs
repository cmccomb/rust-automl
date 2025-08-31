//! Implementation of clustering model training.

use crate::algorithms::ClusteringAlgorithm;
use crate::settings::{ClusteringAlgorithmName, ClusteringSettings};
use smartcore::linalg::basic::arrays::{Array1, Array2};
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};

/// Trains clustering models
pub struct ClusteringModel<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// Settings for the model.
    settings: ClusteringSettings,
    /// Training data.
    x_train: InputArray,
    /// The fitted algorithm.
    algorithm: Option<ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>>,
}

impl<INPUT, CLUSTER, InputArray, ClusterArray>
    ClusteringModel<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// Create a new clustering model.
    pub fn new(x: InputArray, settings: ClusteringSettings) -> Self {
        Self {
            settings,
            x_train: x,
            algorithm: None,
        }
    }

    /// Train the model using the configured algorithm.
    pub fn train(&mut self) {
        let alg = match self.settings.algorithm {
            ClusteringAlgorithmName::KMeans => ClusteringAlgorithm::default_kmeans(),
            ClusteringAlgorithmName::Agglomerative => ClusteringAlgorithm::default_agglomerative(),
            ClusteringAlgorithmName::DBSCAN => ClusteringAlgorithm::default_dbscan(),
        };
        let fitted = alg.fit(&self.x_train, &self.settings);
        self.algorithm = Some(fitted);
    }

    /// Predict cluster assignments for new data.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been trained.
    pub fn predict(&self, x: InputArray) -> ClusterArray {
        match &self.algorithm {
            Some(alg) => alg.predict(&x, &self.settings),
            None => panic!("Model has not been trained"),
        }
    }
}
