#![allow(clippy::needless_doctest_main)]
//! Real-world diabetes progression regression example.
//!
//! The diabetes dataset includes 10 physiological measurements for 442
//! individuals. This example demonstrates how to configure a preprocessing
//! pipeline, tighten algorithm hyperparameters, and evaluate the models via
//! cross-validation before using the best regressor for inference.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example diabetes_regression
//! ```

#[path = "../tests/fixtures/diabetes_dataset.rs"]
mod diabetes_dataset;

use std::error::Error;

use automl::settings::{
    ColumnSelector, FinalAlgorithm, ImputeParams, ImputeStrategy, Kernel, PreprocessingPipeline,
    PreprocessingStep, RandomForestRegressorParameters, RegressionSettings, SVRParameters,
    ScaleParams, ScaleStrategy, StandardizeParams,
};
use automl::{DenseMatrix, RegressionModel};
use diabetes_dataset::load_diabetes_dataset;

fn main() -> Result<(), Box<dyn Error>> {
    let (features, targets) = load_diabetes_dataset()?;

    let preprocessing = PreprocessingPipeline::new()
        .add_step(PreprocessingStep::Impute(ImputeParams {
            strategy: ImputeStrategy::Median,
            selector: ColumnSelector::All,
        }))
        .add_step(PreprocessingStep::Scale(ScaleParams {
            selector: ColumnSelector::All,
            strategy: ScaleStrategy::Standard(StandardizeParams::default()),
        }));

    let settings = RegressionSettings::default()
        .with_number_of_folds(8)
        .shuffle_data(true)
        .with_final_model(FinalAlgorithm::Best)
        .with_preprocessing(preprocessing)
        .with_random_forest_regressor_settings(
            RandomForestRegressorParameters::default()
                .with_n_trees(250)
                .with_max_depth(6)
                .with_min_samples_leaf(2)
                .with_min_samples_split(4),
        )
        .with_svr_settings(
            SVRParameters::default()
                .with_c(12.5)
                .with_eps(0.05)
                .with_kernel(Kernel::RBF(0.35)),
        );

    let mut model = RegressionModel::new(features, targets, settings);
    model.train()?;

    println!("{model}");

    let evaluation_visit = DenseMatrix::from_2d_vec(&vec![vec![
        0.038_075_906,
        0.050_680_119,
        0.061_696_207,
        0.021_872_355,
        -0.044_223_498,
        -0.034_820_763,
        -0.043_400_846,
        -0.002_592_262,
        0.019_908_421,
        -0.017_646_125,
    ]])?;
    let predicted_progression = model.predict(evaluation_visit)?;
    println!("Predicted disease progression: {predicted_progression:?}");

    Ok(())
}
