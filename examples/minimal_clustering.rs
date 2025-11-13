#![allow(clippy::needless_doctest_main)]
//! Minimal clustering example
//!
//! This example demonstrates the minimal steps required to run clustering
//! using the `ClusteringModel` API. It loads a small clustering fixture,
//! builds default clustering settings, trains every available algorithm, and
//! prints predicted cluster assignments for each trained model.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example minimal_clustering
//! ```
use automl::utils::load_csv_features;
use automl::{ClusteringModel, settings::ClusteringSettings};

fn main() {
    // Load some sample data from CSV
    let x = load_csv_features("tests/fixtures/clustering_points.csv").unwrap();

    // Build default clustering settings
    let settings = ClusteringSettings::default().with_k(2);

    // Create and train the model
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();

    // Evaluate clustering quality using the known ground-truth assignments
    // for this fixture dataset.
    let truth = vec![1_u8, 1, 2, 2];
    model.evaluate(&truth);

    // Print trained results
    println!("{model}");

    // Predict cluster assignments for each trained algorithm
    for algorithm in model.trained_algorithm_names() {
        let clusters: Vec<u8> = model
            .predict_with(algorithm, &x)
            .expect("prediction failed");
        println!("{algorithm}: {clusters:?}");
    }

    // Predict cluster assignments
    let clusters: Vec<u8> = model.predict(&x).expect("prediction failed");
    println!("Predicted clusters: {clusters:?}");
}
