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
pub use settings::Settings;

pub mod cookbook;

pub mod utils;

/// Model definitions and implementations.
pub mod model;
pub use model::{ClassificationModel, RegressionModel};

pub use smartcore::linalg::basic::matrix::DenseMatrix;
