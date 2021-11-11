#![warn(clippy::all)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(clippy::missing_docs_in_private_items)]

//! # AutoML with Linfa
//! AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accesible for
//! a general audience. This crate builds on top of the [linfa](https://crates.io/crates/linfa) machine learning framework,
//! and provides some utilities to quickly train and compare models.
//!
//! # Usage
//! For instance, running the following:
//! ```rust
//! fn main() {
//!     let data = linfa_datasets::diabetes();
//!     let r = automl::regression::compare_models(&data);
//!     print!("{}", r);
//! }
//! ```
//! Will output this:
//! ```text
//! ┌──────────────┬───────┬─────────┬─────────┬───────────┐
//! │ Model        │ R^2   │ MSE     │ MAE     │ Exp. Var. │
//! ╞══════════════╪═══════╪═════════╪═════════╪═══════════╡
//! │ Linear Model │ 0.519 │ 2.859e3 │ 4.326e1 │ 5.189e-1  │
//! ├──────────────┼───────┼─────────┼─────────┼───────────┤
//! │ Elastic Net  │ 0.009 │ 5.891e3 │ 6.563e1 │ 8.864e-3  │
//! ├──────────────┼───────┼─────────┼─────────┼───────────┤
//! │ LASSO        │ 0.359 │ 3.811e3 │ 5.254e1 │ 3.589e-1  │
//! ├──────────────┼───────┼─────────┼─────────┼───────────┤
//! │ Ridge        │ 0.007 │ 5.904e3 │ 6.571e1 │ 6.537e-3  │
//! └──────────────┴───────┴─────────┴─────────┴───────────┘
//! ```

// pub mod classification;
// pub mod clustering;
pub mod regression;
