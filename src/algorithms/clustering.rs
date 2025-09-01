//! Clustering algorithm definitions and helpers.

use std::fmt::{Display, Formatter};

use crate::model::{ModelError, ModelResult};
use crate::settings::ClusteringSettings;
use smartcore::cluster::{
    agglomerative::{AgglomerativeClustering, AgglomerativeClusteringParameters},
    dbscan::{DBSCAN, DBSCANParameters},
    kmeans::{KMeans, KMeansParameters},
};
use smartcore::linalg::basic::arrays::{Array1, Array2};
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};

/// Supported clustering algorithms.
pub enum ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// K-Means clustering
    KMeans(Option<KMeans<INPUT, CLUSTER, InputArray, ClusterArray>>),
    /// Agglomerative hierarchical clustering
    Agglomerative(Option<AgglomerativeClustering<INPUT, CLUSTER, InputArray, ClusterArray>>),
    /// DBSCAN clustering
    DBSCAN(Option<DBSCAN<INPUT, CLUSTER, InputArray, ClusterArray, Euclidian<INPUT>>>),
}

impl<INPUT, CLUSTER, InputArray, ClusterArray>
    ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// Default K-Means algorithm
    #[must_use]
    pub const fn default_kmeans() -> Self {
        Self::KMeans(None)
    }

    /// Default Agglomerative algorithm
    #[must_use]
    pub const fn default_agglomerative() -> Self {
        Self::Agglomerative(None)
    }

    /// Default DBSCAN algorithm
    #[must_use]
    pub const fn default_dbscan() -> Self {
        Self::DBSCAN(None)
    }

    /// List all available algorithms
    #[must_use]
    pub fn all_algorithms(_settings: &ClusteringSettings) -> Vec<Self> {
        vec![
            Self::default_kmeans(),
            Self::default_agglomerative(),
            Self::default_dbscan(),
        ]
    }

    /// Fit the algorithm
    pub(crate) fn fit(self, x: &InputArray, settings: &ClusteringSettings) -> Self {
        match self {
            Self::KMeans(_) => {
                let model = KMeans::fit(
                    x,
                    KMeansParameters::default()
                        .with_k(settings.k)
                        .with_max_iter(settings.max_iter),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                );
                Self::KMeans(Some(model))
            }
            Self::Agglomerative(_) => {
                let model = AgglomerativeClustering::fit(
                    x,
                    AgglomerativeClusteringParameters::default().with_n_clusters(settings.k),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                );
                Self::Agglomerative(Some(model))
            }
            Self::DBSCAN(_) => {
                let model = DBSCAN::fit(
                    x,
                    DBSCANParameters::default()
                        .with_eps(settings.eps)
                        .with_min_samples(settings.min_samples),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                );
                Self::DBSCAN(Some(model))
            }
        }
    }

    /// Predict cluster assignments
    pub(crate) fn predict(
        &self,
        x: &InputArray,
        settings: &ClusteringSettings,
    ) -> ModelResult<ClusterArray> {
        match self {
            Self::KMeans(Some(model)) => model
                .predict(x)
                .map_err(|e| ModelError::Inference(e.to_string())),
            Self::Agglomerative(_) => {
                let model = AgglomerativeClustering::<INPUT, usize, InputArray, Vec<usize>>::fit(
                    x,
                    AgglomerativeClusteringParameters::default().with_n_clusters(settings.k),
                )
                .map_err(|e| ModelError::Inference(e.to_string()))?;
                model
                    .labels
                    .into_iter()
                    .map(|l| CLUSTER::from_usize(l).ok_or(ModelError::InvalidClusterLabel(l)))
                    .collect()
            }
            Self::DBSCAN(Some(model)) => model
                .predict(x)
                .map_err(|e| ModelError::Inference(e.to_string())),
            _ => Err(ModelError::NotTrained),
        }
    }
}

impl<INPUT, CLUSTER, InputArray, ClusterArray> Display
    for ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KMeans(_) => write!(f, "KMeans"),
            Self::Agglomerative(_) => write!(f, "Agglomerative"),
            Self::DBSCAN(_) => write!(f, "DBSCAN"),
        }
    }
}
