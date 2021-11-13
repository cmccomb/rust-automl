#![warn(clippy::all)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(clippy::missing_docs_in_private_items)]

//! # AutoML with SmartCore
//! AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accessible for
//! a general audience. This crate builds on top of the [smartcore](https://smartcorelib.org/) machine learning framework,
//! and provides some utilities to quickly train and compare models.
//!
//! # Usage
//! For instance, running the following:
//! ```rust
//! fn main() {
//!    let data = smartcore::dataset::breast_cancer::load_dataset();
//!    let settings = automl::regression::Settings::default();
//!    let r = automl::regression::compare_models(data, settings);
//!    print!("{}", r);
//!}
//! ```
//! Will output this:
//! ```text
//! ┌─────────────────────────┬────────┬─────────┬─────────┐
//! │ Model                   │ R^2    │ MSE     │ MAE     │
//! ╞═════════════════════════╪════════╪═════════╪═════════╡
//! │ LASSO Regressor         │ 0.426  │ 3.360e3 │ 4.654e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Ridge Regressor         │ 0.419  │ 3.399e3 │ 4.688e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Linear Regressor        │ 0.416  │ 3.418e3 │ 4.700e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Elastic Net Regressor   │ 0.384  │ 3.605e3 │ 4.877e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Random Forest Regressor │ 0.327  │ 3.939e3 │ 5.116e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ KNN Regressor           │ 0.300  │ 4.093e3 │ 5.066e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Support Vector Regessor │ 0.086  │ 5.345e3 │ 6.251e1 │
//! ├─────────────────────────┼────────┼─────────┼─────────┤
//! │ Decision Tree Regressor │ -0.050 │ 6.144e3 │ 6.050e1 │
//! └─────────────────────────┴────────┴─────────┴─────────┘
//! ```

pub mod classification;
pub mod regression;
mod traits;
