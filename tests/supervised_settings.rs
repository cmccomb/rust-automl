use automl::DenseMatrix;
use automl::settings::{
    ClassificationSettings, Metric, RegressionSettings, WithSupervisedSettings,
};

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
