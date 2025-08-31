//! Algorithm enumerations and helpers for model training.

pub mod regression;
pub use regression::RegressionAlgorithm;

pub mod classification;
pub use classification::ClassificationAlgorithm;

pub mod clustering;
pub use clustering::ClusteringAlgorithm;
