#![allow(clippy::needless_doctest_main)]
//! Maximal clustering example
//!
//! This example demonstrates advanced configuration for DBSCAN clustering
//! using the `ClusteringModel` API. It loads a small clustering fixture,
//! customizes the settings, trains the algorithm, and prints predicted
//! cluster assignments.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example maximal_clustering
//! ```
#[path = "../tests/fixtures/clustering_data.rs"]
mod clustering_data;

use automl::{
    ClusteringModel, DenseMatrix,
    settings::{ClusteringAlgorithmName, ClusteringSettings},
};
use clustering_data::clustering_testing_data;

fn main() {
    // Load some sample data
    let x = clustering_testing_data();

    // Customize clustering settings for DBSCAN
    let settings = ClusteringSettings::default()
        .with_algorithm(ClusteringAlgorithmName::DBSCAN)
        .with_min_samples(2)
        .verbose(true);

    // Create and train the model
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();

    // Predict cluster assignments for new data
    let new_points = DenseMatrix::from_2d_vec(&vec![vec![0.9_f64, 1.1], vec![8.1, 8.3]]).unwrap();
    let clusters: Vec<u8> = model.predict(&new_points);
    println!("Predicted clusters: {clusters:?}");
}
