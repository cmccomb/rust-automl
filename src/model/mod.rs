//! Types and implementations for machine learning models.

pub mod classification;
mod preprocessing;
pub mod regression;

pub use classification::ClassificationModel;
pub use regression::RegressionModel;
