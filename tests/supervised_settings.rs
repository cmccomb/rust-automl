use automl::DenseMatrix;
use automl::RegressionAlgorithm;
use automl::settings::{ClassificationSettings, Metric, RegressionSettings};

#[test]
fn classification_builder_delegates() {
    let settings = ClassificationSettings::default()
        .with_number_of_folds(5)
        .shuffle_data(true);
    let kfold = settings.get_kfolds();
    assert_eq!(kfold.n_splits, 5);
    assert!(kfold.shuffle);
}

#[test]
fn regression_builder_delegates() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .with_number_of_folds(4)
        .shuffle_data(false)
        .sorted_by(Metric::MeanAbsoluteError);
    let kfold = settings.get_kfolds();
    assert_eq!(kfold.n_splits, 4);
    assert!(!kfold.shuffle);
}

#[test]
fn regression_only_is_idempotent() {
    let linear = RegressionAlgorithm::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default_linear();
    let settings = RegressionSettings::default().only(&linear).only(&linear);
    assert!(RegressionAlgorithm::all_algorithms(&settings) == vec![linear]);
}

#[test]
fn regression_only_replaces_previous_selection_and_skips() {
    let linear = RegressionAlgorithm::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default_linear();
    let tree = RegressionAlgorithm::default_decision_tree();
    let settings = RegressionSettings::default().only(&linear).only(&tree);
    assert!(RegressionAlgorithm::all_algorithms(&settings) == vec![tree]);

    let settings = RegressionSettings::default()
        .skip(RegressionAlgorithm::default_linear())
        .only(&linear);
    assert!(RegressionAlgorithm::all_algorithms(&settings) == vec![linear]);
}
