//! Types and implementations for machine learning models.

pub mod classification;
pub mod clustering;
mod comparison;
pub mod error;
pub mod preprocessing;
pub mod regression;
pub mod supervised;

pub use classification::ClassificationModel;
pub use clustering::ClusteringModel;
pub use comparison::ComparisonEntry;
pub use error::{ModelError, ModelResult};
pub use regression::RegressionModel;
pub use supervised::{Algorithm, SupervisedLearningSettings, SupervisedModel};
