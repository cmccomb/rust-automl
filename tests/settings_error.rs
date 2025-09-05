use automl::settings::{ClassificationSettings, Metric, SettingsError};
use automl::{DenseMatrix, RegressionSettings};

#[test]
fn classification_metric_not_set_returns_error() {
    let settings = ClassificationSettings::default().sorted_by(Metric::None);
    let err = settings.get_metric::<u32, Vec<u32>>().unwrap_err();
    assert_eq!(err, SettingsError::MetricNotSet);
}

#[test]
fn classification_metric_unsupported_returns_error() {
    let settings = ClassificationSettings::default().sorted_by(Metric::RSquared);
    let err = settings.get_metric::<u32, Vec<u32>>().unwrap_err();
    assert_eq!(err, SettingsError::UnsupportedMetric(Metric::RSquared));
}

#[test]
fn regression_metric_not_set_returns_error() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .sorted_by(Metric::None);
    let err = settings.get_metric().unwrap_err();
    assert_eq!(err, SettingsError::MetricNotSet);
}

#[test]
fn regression_metric_unsupported_returns_error() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .sorted_by(Metric::Accuracy);
    let err = settings.get_metric().unwrap_err();
    assert_eq!(err, SettingsError::UnsupportedMetric(Metric::Accuracy));
}
