#[path = "fixtures/regression_data.rs"]
mod regression_data;

use automl::algorithms::RegressionAlgorithm;
use automl::model::Algorithm;
use automl::settings::{Distance, KNNParameters, Kernel, SVRParameters, XGRegressorParameters};
use automl::{DenseMatrix, RegressionSettings, SupervisedModel};
use regression_data::regression_testing_data;
use smartcore::error::FailedError;

#[test]
fn test_default_regression() {
    let settings = RegressionSettings::default();
    test_from_settings(settings);
}

#[test]
fn test_knn_only_regression() {
    for distance in [
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::Minkowski(3),
        Distance::Hamming,
    ] {
        let settings = RegressionSettings::default()
            .with_knn_regressor_settings(KNNParameters::default().with_distance(distance))
            .only(&RegressionAlgorithm::default_knn_regressor());
        test_from_settings(settings);
    }
}

#[test]
fn test_svr_regression_multiple_kernels() {
    let kernels = vec![
        Kernel::Linear,
        Kernel::RBF(0.25),
        Kernel::Polynomial(3.0, 0.5, 1.0),
        Kernel::Sigmoid(0.1, 0.0),
    ];
    for kernel in kernels {
        let settings = RegressionSettings::default()
            .with_svr_settings(
                SVRParameters::default()
                    .with_eps(0.2)
                    .with_tol(5e-4)
                    .with_c(1.2)
                    .with_kernel(kernel),
            )
            .only(&RegressionAlgorithm::default_support_vector_regressor());
        test_from_settings(settings);
    }
}

#[test]
fn test_svr_skiplist_controls_algorithms() {
    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default().skip(RegressionAlgorithm::default_support_vector_regressor());
    let algorithms = RegressionAlgorithm::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .all(|algo| !matches!(algo, RegressionAlgorithm::SupportVectorRegressor(_)))
    );

    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default()
            .only(&RegressionAlgorithm::default_support_vector_regressor());
    let algorithms = RegressionAlgorithm::all_algorithms(&settings);
    assert_eq!(algorithms.len(), 1);
    assert!(matches!(
        algorithms[0],
        RegressionAlgorithm::SupportVectorRegressor(_)
    ));
}

#[test]
fn test_svr_missing_settings_error() {
    let (x, y) = regression_testing_data();
    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default().without_svr_settings();
    let algo = RegressionAlgorithm::default_support_vector_regressor();
    let err = algo
        .fit(&x, &y, &settings)
        .err()
        .expect("expected missing SVR settings to error");
    assert_eq!(err.error(), FailedError::ParametersError);
}

#[test]
fn test_xgboost_regression_trains_successfully() {
    let (x, y) = regression_testing_data();
    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default()
            .with_xgboost_settings(
                XGRegressorParameters::default()
                    .with_n_estimators(5)
                    .with_learning_rate(0.2)
                    .with_max_depth(3)
                    .with_subsample(0.75),
            )
            .only(&RegressionAlgorithm::default_xgboost_regressor());
    let algo = RegressionAlgorithm::default_xgboost_regressor();
    let trained = algo
        .fit(&x, &y, &settings)
        .expect("xgboost should train successfully");
    let predictions = trained
        .predict(&x)
        .expect("xgboost predictions should succeed");
    assert_eq!(predictions.len(), y.len());
    RegressionAlgorithm::default_xgboost_regressor()
        .cv(&x, &y, &settings)
        .expect("xgboost cross-validation should succeed");
}

#[test]
fn test_xgboost_invalid_parameters_error() {
    let (x, y) = regression_testing_data();
    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default().with_xgboost_settings(
            XGRegressorParameters::default()
                .with_learning_rate(0.0)
                .with_n_estimators(10),
        );
    let algo = RegressionAlgorithm::default_xgboost_regressor();
    let err = algo
        .fit(&x, &y, &settings)
        .err()
        .expect("invalid learning rate should fail");
    assert_eq!(err.error(), FailedError::ParametersError);
    assert!(
        err.to_string()
            .contains("xgboost learning rate must be greater than zero")
    );

    let cv_err = RegressionAlgorithm::default_xgboost_regressor()
        .cv(&x, &y, &settings)
        .err()
        .expect("invalid learning rate should fail in cross-validation");
    assert_eq!(cv_err.error(), FailedError::ParametersError);
    assert!(
        cv_err
            .to_string()
            .contains("xgboost learning rate must be greater than zero")
    );
}

#[test]
fn test_xgboost_skiplist_controls_algorithms() {
    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default().skip(RegressionAlgorithm::default_xgboost_regressor());
    let algorithms = RegressionAlgorithm::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .all(|algo| !matches!(algo, RegressionAlgorithm::XGBoostRegressor(_)))
    );

    let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        RegressionSettings::default().only(&RegressionAlgorithm::default_xgboost_regressor());
    let algorithms = RegressionAlgorithm::all_algorithms(&settings);
    assert_eq!(algorithms.len(), 1);
    assert!(matches!(
        algorithms[0],
        RegressionAlgorithm::XGBoostRegressor(_)
    ));
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
