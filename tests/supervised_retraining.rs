use automl::settings::{LassoParameters, Metric, PreprocessingStep, StandardizeParams};
use automl::{
    ClassificationModel, ClassificationSettings, DenseMatrix, ModelError, RegressionAlgorithm,
    RegressionModel, RegressionSettings,
};
use smartcore::error::FailedError;

type Regressor = RegressionModel<f64, f64, DenseMatrix<f64>, Vec<f64>>;
type Settings = RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>;

fn data() -> (DenseMatrix<f64>, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..24).map(|i| vec![f64::from(i)]).collect();
    let targets = (0..24).map(|i| 2.0 * f64::from(i) + 1.0).collect();
    (DenseMatrix::from_2d_vec(&rows).unwrap(), targets)
}

fn query() -> DenseMatrix<f64> {
    DenseMatrix::from_2d_array(&[&[100.0]]).unwrap()
}

fn linear_settings() -> Settings {
    Settings::default().only(&RegressionAlgorithm::default_linear())
}

fn trained_linear() -> Regressor {
    let (x, y) = data();
    let mut model = Regressor::new(x, y, linear_settings());
    model.train().unwrap();
    model
}

#[test]
fn repeated_training_replaces_regression_leaderboard() {
    let mut model = trained_linear();
    model.train().unwrap();
    assert_eq!(format!("{model}").matches("Linear Regressor").count(), 1);
}

#[test]
fn retraining_uses_only_newly_selected_algorithm() {
    let mut model = trained_linear();
    model.settings = Settings::default().only(&RegressionAlgorithm::default_decision_tree());
    model.train().unwrap();

    let (x, y) = data();
    let mut fresh = Regressor::new(
        x,
        y,
        Settings::default().only(&RegressionAlgorithm::default_decision_tree()),
    );
    fresh.train().unwrap();

    assert_eq!(
        model.predict(query()).unwrap(),
        fresh.predict(query()).unwrap()
    );
    let table = format!("{model}");
    assert!(!table.contains("Linear Regressor"));
    assert_eq!(table.matches("Decision Tree Regressor").count(), 1);
}

#[test]
fn retraining_replaces_preprocessing_and_metric_together() {
    let mut model = trained_linear();
    model.settings = linear_settings()
        .add_step(PreprocessingStep::Standardize(StandardizeParams::default()))
        .sorted_by(Metric::MeanAbsoluteError);
    model.train().unwrap();

    assert!((model.predict(query()).unwrap()[0] - 201.0).abs() < 1e-8);
    let table = format!("{model}");
    assert_eq!(table.matches("Linear Regressor").count(), 1);
    assert!(table.contains("Testing MAE"));
    assert!(!table.contains("Testing R^2"));

    model.settings = linear_settings();
    model.train().unwrap();
    assert!((model.predict(query()).unwrap()[0] - 201.0).abs() < 1e-8);
    assert_eq!(format!("{model}").matches("Linear Regressor").count(), 1);
}

#[test]
fn preprocessing_failure_preserves_previous_model() {
    let mut model = trained_linear();
    let before_prediction = model.predict(query()).unwrap();
    let before_table = format!("{model}");
    model.settings = linear_settings()
        .add_step(PreprocessingStep::Standardize(StandardizeParams::default()))
        .add_step(PreprocessingStep::ReplaceWithPCA {
            number_of_components: 2,
        });

    assert!(model.train().is_err());
    assert_eq!(model.predict(query()).unwrap(), before_prediction);
    assert_eq!(format!("{model}"), before_table);
}

#[test]
fn later_algorithm_failure_preserves_previous_model_and_scores() {
    let mut model = trained_linear();
    let before_prediction = model.predict(query()).unwrap();
    let before_table = format!("{model}");
    // Linear and ridge regression succeed, then lasso fails. Neither the new linear model
    // nor its fitted scaler should replace the previous successful run.
    model.settings = Settings::default()
        .with_lasso_settings(LassoParameters::default().with_alpha(-1.0))
        .add_step(PreprocessingStep::Standardize(StandardizeParams::default()))
        .sorted_by(Metric::MeanAbsoluteError);

    assert!(model.train().is_err());
    assert_eq!(model.predict(query()).unwrap(), before_prediction);
    assert_eq!(format!("{model}"), before_table);

    model.settings = linear_settings();
    model.train().unwrap();
    assert_eq!(format!("{model}").matches("Linear Regressor").count(), 1);
}

#[test]
fn failed_initial_training_does_not_expose_partial_models() {
    let (x, y) = data();
    let mut model = Regressor::new(
        x,
        y,
        Settings::default().with_lasso_settings(LassoParameters::default().with_alpha(-1.0)),
    );

    assert!(model.train().is_err());
    assert_eq!(model.predict(query()).unwrap_err(), ModelError::NotTrained);
    assert!(!format!("{model}").contains("Linear Regressor"));
}

#[test]
fn repeated_training_replaces_classification_leaderboard() {
    let (x, _) = data();
    let y: Vec<u32> = (0..24).map(|i| u32::from(i >= 12)).collect();
    let mut model: ClassificationModel<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationModel::new(x, y, ClassificationSettings::default());
    model.train().unwrap();
    model.train().unwrap();

    let table = format!("{model}");
    assert_eq!(table.matches("Decision Tree Classifier").count(), 1);
    assert_eq!(table.matches("KNN Classifier").count(), 1);
}

#[test]
fn empty_algorithm_selection_returns_error_and_preserves_model() {
    let mut model = trained_linear();
    let before_table = format!("{model}");
    let before_prediction = model.predict(query()).unwrap();
    model.settings = linear_settings().skip(RegressionAlgorithm::default_linear());

    let err = model.train().unwrap_err();
    assert_eq!(err.error(), FailedError::ParametersError);
    assert!(err.to_string().contains("no algorithms"));
    assert_eq!(format!("{model}"), before_table);
    assert_eq!(model.predict(query()).unwrap(), before_prediction);
}

#[test]
fn invalid_fold_counts_return_errors_without_damaging_model() {
    let mut model = trained_linear();
    let before_table = format!("{model}");
    let before_prediction = model.predict(query()).unwrap();

    for folds in [0, 1, 25] {
        model.settings = linear_settings().with_number_of_folds(folds);
        let err = model.train().unwrap_err();
        assert_eq!(err.error(), FailedError::ParametersError);
        assert!(err.to_string().contains("folds"));
        assert_eq!(format!("{model}"), before_table);
        assert_eq!(model.predict(query()).unwrap(), before_prediction);
    }
}
