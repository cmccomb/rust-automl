#![deny(clippy::correctness)]
#![warn(
    clippy::all,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style
)]
#![allow(clippy::module_name_repetitions)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod settings;
pub use settings::{ClassificationSettings, ClusteringSettings, RegressionSettings};

pub mod cookbook;

pub mod utils;

/// Metric re-exports.
pub mod metrics;

/// Algorithm enumerations and helpers.
pub mod algorithms;
pub use algorithms::{ClassificationAlgorithm, ClusteringAlgorithm, RegressionAlgorithm};

/// Model definitions and implementations.
pub mod model;
pub use model::{ClassificationModel, ClusteringModel, ModelError, ModelResult, RegressionModel};

pub use smartcore::linalg::basic::matrix::DenseMatrix;
