#![allow(clippy::needless_doctest_main)]
//! Real-world breast cancer classification example.
//!
//! This example trains a multi-model comparison on the Wisconsin Diagnostic
//! Breast Cancer dataset bundled with the repository. The workflow shows how to
//! load a CSV file, wire up a preprocessing pipeline, and customize the
//! algorithms that participate in the comparison.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example breast_cancer_csv
//! ```

#[path = "../tests/fixtures/breast_cancer_dataset.rs"]
mod breast_cancer_dataset;

use std::error::Error;

use automl::settings::{
    ClassificationSettings, FinalAlgorithm, PreprocessingPipeline, PreprocessingStep,
    RandomForestClassifierParameters, StandardizeParams,
};
use automl::{ClassificationModel, DenseMatrix};
use breast_cancer_dataset::load_breast_cancer_dataset;

fn main() -> Result<(), Box<dyn Error>> {
    let (features, targets) = load_breast_cancer_dataset()?;

    let preprocessing = PreprocessingPipeline::new()
        .add_step(PreprocessingStep::Standardize(StandardizeParams::default()));

    let settings = ClassificationSettings::default()
        .with_number_of_folds(5)
        .shuffle_data(true)
        .with_final_model(FinalAlgorithm::Best)
        .with_preprocessing(preprocessing)
        .with_random_forest_classifier_settings(
            RandomForestClassifierParameters::default()
                .with_n_trees(200)
                .with_max_depth(8)
                .with_min_samples_split(4)
                .with_min_samples_leaf(2),
        );

    let mut model = ClassificationModel::new(features, targets, settings);
    model.train()?;

    println!("{model}");

    let example_patient = DenseMatrix::from_2d_vec(&vec![vec![
        13.540, 14.360, 87.460, 566.300, 0.097, 0.052, 0.024, 0.015, 0.153, 0.055, 0.284, 0.915,
        2.376, 23.420, 0.005, 0.013, 0.010, 0.005, 0.018, 0.002, 14.230, 17.730, 91.760, 618.800,
        0.118, 0.115, 0.068, 0.025, 0.210, 0.062,
    ]])?;
    let predictions = model.predict(example_patient)?;
    println!("Predicted class for the evaluation patient: {predictions:?}");

    Ok(())
}
