#![allow(clippy::needless_doctest_main)]
//! Maximal classification example
//!
//! This example demonstrates the maximal steps required to run a model
//! comparison using the `ClassificationModel` API. It loads a small classification
//! fixture, builds custom classification settings, trains all configured
//! algorithms using cross-validation, and prints a comparison table.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example maximal_classification
//! ```

#[path = "../tests/fixtures/classification_data.rs"]
mod classification_data;

use automl::settings::ClassificationSettings;
use automl::settings::{
    DecisionTreeClassifierParameters, Distance, FinalAlgorithm, KNNAlgorithmName, KNNParameters,
    KNNWeightFunction,
};
use automl::{ClassificationModel, DenseMatrix};
use classification_data::classification_testing_data;

fn main() {
    // Load some classification data
    let (x, y) = classification_testing_data();

    // Customize settings
    let settings = ClassificationSettings::default()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .with_final_model(FinalAlgorithm::Best)
        .with_knn_classifier_settings(
            KNNParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(3)
                .with_distance(Distance::Euclidean)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .with_decision_tree_classifier_settings(
            DecisionTreeClassifierParameters::default()
                .with_min_samples_split(2)
                .with_max_depth(15)
                .with_min_samples_leaf(1),
        );

    // Load a dataset and add it to the classifier along with the customized settings
    let mut model = ClassificationModel::new(x, y, settings);

    // Run a model comparison with all models at customized settings
    model.train().unwrap();

    // Display comparison results
    println!("{model}");

    // Predict with the model, be sure to use a DenseMatrix
    let preds = model
        .predict(DenseMatrix::from_2d_vec(&vec![vec![0.5_f64, 0.5]; 5]).unwrap())
        .expect("prediction should succeed");
    println!("Predictions: {preds:?}");
}
