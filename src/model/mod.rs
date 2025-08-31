//! Types and implementations for machine learning models.

pub mod classification;
pub mod clustering;
mod comparison;
mod preprocessing;
pub mod regression;

pub use classification::ClassificationModel;
pub use clustering::ClusteringModel;
pub use comparison::ComparisonEntry;
pub use regression::RegressionModel;
