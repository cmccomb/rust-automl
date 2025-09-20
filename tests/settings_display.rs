use automl::algorithms::RegressionAlgorithm;
use automl::settings::{
    ClassificationSettings, DecisionTreeClassifierParameters, KNNParameters,
    LinearRegressionParameters, LogisticRegressionParameters, MultinomialNBParameters,
    RandomForestClassifierParameters, RandomForestRegressorParameters,
};
use automl::{DenseMatrix, RegressionSettings};

#[test]
fn display_reflects_skiplist() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .with_linear_settings(LinearRegressionParameters::default())
        .skip(RegressionAlgorithm::default_random_forest());
    let output = format!("{settings}");
    assert!(output.contains("Regression settings"));
}

#[test]
fn random_forest_section_labels_n_trees() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .with_random_forest_regressor_settings(RandomForestRegressorParameters::default());
    let output = format!("{settings}");
    assert!(output.contains("Regression settings"));
}

#[test]
fn classification_builder_methods_chain() {
    let settings = ClassificationSettings::default()
        .with_knn_classifier_settings(KNNParameters::default())
        .with_decision_tree_classifier_settings(DecisionTreeClassifierParameters::default())
        .with_random_forest_classifier_settings(RandomForestClassifierParameters::default())
        .with_logistic_regression_settings(LogisticRegressionParameters::default())
        .with_multinomial_nb_settings(MultinomialNBParameters::default())
        .with_number_of_folds(2);
    let folds = settings.get_kfolds();
    assert_eq!(folds.n_splits, 2);
}
