#[path = "fixtures/regression_data.rs"]
mod regression_data;

use automl::algorithms::RegressionAlgorithm;
use automl::settings::{
    KNNAlgorithmName, KNNRegressorParameters, KNNWeightFunction, build_knn_regressor_parameters,
};
use automl::{DenseMatrix, RegressionModel, RegressionSettings};
use regression_data::regression_testing_data;
use smartcore::metrics::distance::manhattan::Manhattan;

#[test]
fn test_default_regression() {
    let settings = RegressionSettings::default();
    test_from_settings(settings);
}

#[test]
fn test_knn_only_regression() {
    let settings =
        RegressionSettings::default().only(&RegressionAlgorithm::default_knn_regressor());
    test_from_settings(settings);
}

#[test]
fn test_build_knn_regressor_parameters_helper() {
    let settings = KNNRegressorParameters::default()
        .with_k(3)
        .with_algorithm(KNNAlgorithmName::CoverTree)
        .with_weight(KNNWeightFunction::Uniform);
    let params = build_knn_regressor_parameters::<f64, _>(&settings, Manhattan::new());
    assert_eq!(params.k, 3);
    assert!(matches!(params.algorithm, KNNAlgorithmName::CoverTree));
    assert!(matches!(params.weight, KNNWeightFunction::Uniform));
}

fn test_from_settings(settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>) {
    // Get test data
    let (x, y) = regression_testing_data();

    // Set up the regressor settings and load data
    let mut regressor = RegressionModel::new(x, y, settings);

    // Compare models
    regressor.train();

    // Try to predict something
    regressor.predict(
        DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
        ])
        .unwrap(),
    );
}
