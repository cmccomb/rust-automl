#![allow(clippy::needless_doctest_main)]
//! Minimal regression example
//!
//! This example demonstrates the minimal steps required to run a model
//! comparison using the `RegressionModel` API. It loads a small regression
//! fixture, builds default regression settings, trains all configured
//! algorithms using cross-validation, and prints a comparison table.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example minimal_regression
//! ```

#[path = "../tests/fixtures/regression_data.rs"]
mod regression_data;

use automl::{RegressionModel, RegressionSettings};
use regression_data::regression_testing_data;
use smartcore::error::Failed;

fn main() -> Result<(), Failed> {
    // Load some regression data
    let (x, y) = regression_testing_data();

    // Totally customize settings
    let settings = RegressionSettings::default();

    // Load a dataset from smartcore and add it to the regressor along with the customized settings
    let mut model = RegressionModel::new(x, y, settings);

    // Run a model comparison with all models at default settings
    model.train()?;
    Ok(())
}
