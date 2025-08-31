#![deny(clippy::correctness)]
#![warn(
    clippy::all,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items
)]
#![allow(clippy::module_name_repetitions, clippy::too_many_lines)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod settings;
pub use settings::Settings;

pub mod cookbook;

/// Supervised learning utilities.
pub mod supervised_model;

mod utils;
pub use utils::regression_testing_data;

pub use smartcore::linalg::basic::matrix::DenseMatrix;
