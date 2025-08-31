#![allow(clippy::needless_doctest_main)]
//! Minimal clustering example
//!
//! This example demonstrates the minimal steps required to run K-Means
//! clustering using the `ClusteringModel` API. It loads a small clustering
//! fixture, builds default clustering settings, trains the algorithm, and
//! prints predicted cluster assignments.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example minimal_clustering
//! ```
#[path = "../tests/fixtures/clustering_data.rs"]
mod clustering_data;

use automl::{ClusteringModel, settings::ClusteringSettings};
use clustering_data::clustering_testing_data;

fn main() {
    // Load some sample data
    let x = clustering_testing_data();

    // Build default clustering settings
    let settings = ClusteringSettings::default().with_k(2);

    // Create and train the model
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();

    // Predict cluster assignments
    let clusters: Vec<u8> = model.predict(x);
    println!("Predicted clusters: {clusters:?}");
}
