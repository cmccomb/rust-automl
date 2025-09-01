#[path = "fixtures/regression_data.rs"]
mod regression_data;

use automl::algorithms::RegressionAlgorithm;
use automl::{DenseMatrix, RegressionSettings, SupervisedModel};
use regression_data::regression_testing_data;

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

fn test_from_settings(settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>) {
    // Set up the regressor settings and load data
    type Model = SupervisedModel<
        RegressionAlgorithm<f64, f64, DenseMatrix<f64>, Vec<f64>>,
        RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>,
        DenseMatrix<f64>,
        Vec<f64>,
    >;

    // Get test data
    let (x, y) = regression_testing_data();

    let mut regressor: Model = SupervisedModel::new(x, y, settings);

    regressor.train().unwrap();

    let table = format!("{regressor}");
    assert!(table.contains("Model"));

    regressor
        .predict(
            DenseMatrix::from_2d_array(&[
                &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
                &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            ])
            .unwrap(),
        )
        .expect("prediction should succeed");
}
