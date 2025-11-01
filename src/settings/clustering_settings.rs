//! Settings for clustering models

use std::fmt::{Display, Formatter};

/// Available clustering algorithms.
#[derive(Clone, Copy)]
pub enum ClusteringAlgorithmName {
    /// K-Means clustering
    KMeans,
    /// Agglomerative hierarchical clustering
    Agglomerative,
    /// DBSCAN density-based clustering
    DBSCAN,
}

/// Settings for clustering algorithms such as K-Means, Agglomerative, or
/// DBSCAN.
///
/// # Examples
/// ```
/// use automl::settings::{ClusteringAlgorithmName, ClusteringSettings};
/// let settings =
///     ClusteringSettings::default().with_algorithm(ClusteringAlgorithmName::DBSCAN);
/// ```
#[derive(Clone)]
pub struct ClusteringSettings {
    /// Number of clusters to produce (for algorithms that require it)
    pub(crate) k: usize,
    /// Maximum number of iterations (used by K-Means)
    pub(crate) max_iter: usize,
    /// DBSCAN neighborhood radius
    pub(crate) eps: f64,
    /// DBSCAN minimum samples per core point
    pub(crate) min_samples: usize,
    /// Selected clustering algorithm
    pub(crate) algorithm: ClusteringAlgorithmName,
    /// Verbosity flag
    pub(crate) verbose: bool,
}

impl Default for ClusteringSettings {
    fn default() -> Self {
        Self {
            k: 2,
            max_iter: 100,
            eps: 0.5,
            min_samples: 5,
            algorithm: ClusteringAlgorithmName::KMeans,
            verbose: false,
        }
    }
}

impl ClusteringSettings {
    /// Set number of clusters
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set maximum iterations
    #[must_use]
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the DBSCAN neighborhood radius
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the minimum samples per core point for DBSCAN
    #[must_use]
    pub const fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Choose the clustering algorithm
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: ClusteringAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Enable verbose logging
    #[must_use]
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Display for ClusteringAlgorithmName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KMeans => write!(f, "KMeans"),
            Self::Agglomerative => write!(f, "Agglomerative"),
            Self::DBSCAN => write!(f, "DBSCAN"),
        }
    }
}
