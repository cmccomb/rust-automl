#![allow(clippy::needless_doctest_main)]
//! Minimal classification example
//!
//! This example demonstrates the minimal steps required to run a model
//! comparison using the `ClassificationModel` API. It loads a small classification
//! fixture, builds default classification settings, trains all configured
//! algorithms using cross-validation.
//!
//! If your features are non-negative integer counts you can opt into
//! Multinomial Naive Bayes with
//! `ClassificationSettings::default().with_multinomial_nb_settings(...)`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example minimal_classification
//! ```

#[path = "../tests/fixtures/classification_data.rs"]
mod classification_data;

use automl::ClassificationModel;
use automl::settings::ClassificationSettings;
use classification_data::classification_testing_data;

fn main() {
    // Load some classification data
    let (x, y) = classification_testing_data();

    // Build default classification settings
    let settings = ClassificationSettings::default();

    // Load a dataset from smartcore and add it to the classifier along with the settings
    let mut model = ClassificationModel::new(x, y, settings);

    // Train the model
    model.train().unwrap();

    // Run a model comparison with all models at default settings
    println!("{model}")
}
