#![allow(clippy::needless_doctest_main)]
//! Maximal regression example
//!
//! This example demonstrates the maximal steps required to run a model
//! comparison using the `RegressionModel` API. It loads a small regression
//! fixture, builds default regression settings, trains all configured
//! algorithms using cross-validation, and prints a comparison table.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example maximal_regression
//! ```

#[path = "../tests/fixtures/regression_data.rs"]
mod regression_data;

use automl::{
    DenseMatrix, RegressionModel, RegressionSettings,
    algorithms::RegressionAlgorithm,
    settings::{
        DecisionTreeRegressorParameters, Distance, ElasticNetParameters, FinalAlgorithm,
        KNNAlgorithmName, KNNParameters, KNNWeightFunction, Kernel, LassoParameters,
        LinearRegressionParameters, LinearRegressionSolverName, Metric,
        RandomForestRegressorParameters, RidgeRegressionParameters, RidgeRegressionSolverName,
        SVRParameters,
    },
};
use regression_data::regression_testing_data;
use smartcore::error::Failed;

fn main() -> Result<(), Failed> {
    // Load some regression data
    let (x, y) = regression_testing_data();

    // Totally customize settings
    let settings = RegressionSettings::default()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .with_final_model(FinalAlgorithm::Best)
        .skip(RegressionAlgorithm::default_random_forest())
        .sorted_by(Metric::RSquared)
        // .with_preprocessing(PreProcessing::AddInteractions)
        .with_linear_settings(
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
        )
        .with_lasso_settings(
            LassoParameters::default()
                .with_alpha(1.0)
                .with_tol(1e-4)
                .with_normalize(true)
                .with_max_iter(1000),
        )
        .with_ridge_settings(
            RidgeRegressionParameters::default()
                .with_alpha(1.0)
                .with_normalize(true)
                .with_solver(RidgeRegressionSolverName::Cholesky),
        )
        .with_elastic_net_settings(
            ElasticNetParameters::default()
                .with_tol(1e-4)
                .with_normalize(true)
                .with_alpha(1.0)
                .with_max_iter(1000)
                .with_l1_ratio(0.5),
        )
        .with_knn_regressor_settings(
            KNNParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(3)
                .with_distance(Distance::Euclidean)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .with_svr_settings(
            SVRParameters::default()
                .with_eps(0.05)
                .with_tol(1e-4)
                .with_c(2.5)
                .with_kernel(Kernel::RBF(0.4)),
        )
        .with_random_forest_regressor_settings(
            RandomForestRegressorParameters::default()
                .with_m(1)
                .with_max_depth(5)
                .with_min_samples_leaf(1)
                .with_n_trees(10)
                .with_min_samples_split(2),
        )
        .with_decision_tree_regressor_settings(
            DecisionTreeRegressorParameters::default()
                .with_min_samples_split(2)
                .with_max_depth(15)
                .with_min_samples_leaf(1),
        );

    // Load a dataset from smartcore and add it to the regressor along with the customized settings
    let mut model = RegressionModel::new(x, y, settings);

    // Run a model comparison with all models at default settings
    model.train()?;

    // Display comparison results
    println!("{model}");

    // Predict with the model, be sure to use a DenseMatrix
    let preds = model
        .predict(DenseMatrix::from_2d_vec(&vec![vec![5.0_f64; 6]; 10])?)
        .expect("prediction should succeed");
    println!("Predictions: {preds:?}");
    Ok(())
}
