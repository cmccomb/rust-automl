//! Types and implementations for machine learning models.

pub mod classification;
mod comparison;
mod preprocessing;
pub mod regression;

pub use classification::ClassificationModel;
pub use comparison::ComparisonEntry;
pub use regression::RegressionModel;
