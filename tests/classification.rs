#[path = "fixtures/classification_data.rs"]
mod classification_data;

use automl::algorithms::ClassificationAlgorithm;
use automl::model::Algorithm;
use automl::settings::{
    BernoulliNBParameters, CategoricalNBParameters, ClassificationSettings,
    MultinomialNBParameters, RandomForestClassifierParameters, SVCParameters,
};
use automl::{DenseMatrix, ModelError, SupervisedModel};
use classification_data::{
    bernoulli_binary_classification_data, bernoulli_threshold_classification_data,
    classification_testing_data,
};
use smartcore::api::SupervisedEstimator;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;

#[test]
fn test_default_classification() {
    let settings = ClassificationSettings::default()
        .with_number_of_folds(3)
        .with_random_forest_classifier_settings(
            RandomForestClassifierParameters::default().with_n_trees(10),
        );
    test_from_settings(settings);
}

#[test]
fn test_all_algorithms_included() {
    let settings = ClassificationSettings::default();
    let algorithms = <ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> as
        automl::model::Algorithm<ClassificationSettings>>::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::RandomForestClassifier(_)))
    );
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::LogisticRegression(_)))
    );
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::GaussianNB(_)))
    );
    assert!(
        algorithms
            .iter()
            .all(|a| !matches!(a, ClassificationAlgorithm::SupportVectorClassifier(_)))
    );
    assert!(
        algorithms
            .iter()
            .all(|a| !matches!(a, ClassificationAlgorithm::MultinomialNB(_)))
    );
    assert!(
        algorithms
            .iter()
            .all(|a| !matches!(a, ClassificationAlgorithm::BernoulliNB(_)))
    );
}

#[test]
fn categorical_nb_algorithm_available_when_enabled() {
    let settings = ClassificationSettings::default()
        .with_categorical_nb_settings(CategoricalNBParameters::default());
    let algorithms = <ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> as
        automl::model::Algorithm<ClassificationSettings>>::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::CategoricalNB(_)))
    );
}

#[test]
fn multinomial_nb_algorithm_available_when_enabled() {
    let settings = ClassificationSettings::default()
        .with_multinomial_nb_settings(MultinomialNBParameters::default());
    let algorithms = <ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> as
        automl::model::Algorithm<ClassificationSettings>>::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::MultinomialNB(_)))
    );
}

#[test]
fn bernoulli_nb_algorithm_available_when_enabled() {
    let settings = ClassificationSettings::default()
        .with_bernoulli_nb_settings(BernoulliNBParameters::default());
    let algorithms = <ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> as
        automl::model::Algorithm<ClassificationSettings>>::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::BernoulliNB(_)))
    );
}

#[test]
fn support_vector_classifier_algorithm_available_when_enabled() {
    let settings = ClassificationSettings::default().with_svc_settings(SVCParameters::default());
    let algorithms = <ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> as
        automl::model::Algorithm<ClassificationSettings>>::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, ClassificationAlgorithm::SupportVectorClassifier(_)))
    );
}

fn test_from_settings(settings: ClassificationSettings) {
    type Model = SupervisedModel<
        ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>>,
        ClassificationSettings,
        DenseMatrix<f64>,
        Vec<u32>,
    >;

    let (x, y) = classification_testing_data();
    let mut model: Model = SupervisedModel::new(x, y, settings);
    model.train().unwrap();

    let table = format!("{model}");
    assert!(table.contains("Model"));

    model
        .predict(DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap())
        .expect("prediction should succeed");
}

#[test]
fn predict_requires_training() {
    type Model = SupervisedModel<
        ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>>,
        ClassificationSettings,
        DenseMatrix<f64>,
        Vec<u32>,
    >;
    let (x, y) = classification_testing_data();
    let model: Model = SupervisedModel::new(x, y, ClassificationSettings::default());
    let res = model.predict(DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap());
    assert!(matches!(res, Err(ModelError::NotTrained)));
}

#[test]
fn support_vector_classifier_trains_and_predicts() {
    let (x, y) = classification_testing_data();
    let settings = ClassificationSettings::default().with_svc_settings(SVCParameters::default());
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_support_vector_classifier();
    let trained = algorithm
        .fit(&x, &y, &settings)
        .expect("SVC training should succeed");
    let predictions = trained.predict(&x).expect("SVC prediction should succeed");
    assert_eq!(predictions.len(), y.len());
}

#[test]
fn bernoulli_nb_trains_on_binary_features() {
    let (x, y) = bernoulli_binary_classification_data();
    let params = BernoulliNBParameters::<f64> {
        binarize: None,
        ..BernoulliNBParameters::default()
    };
    let settings = ClassificationSettings::default().with_bernoulli_nb_settings(params);
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_bernoulli_nb();
    let trained = algorithm
        .fit(&x, &y, &settings)
        .expect("Bernoulli NB training should succeed on binary data");
    let predictions = trained
        .predict(&x)
        .expect("Bernoulli NB prediction should succeed on binary data");
    assert_eq!(predictions.len(), y.len());
    assert!(predictions.iter().all(|&value| value <= 1));
}

#[test]
fn bernoulli_nb_binarizes_with_threshold() {
    let (x, y) = bernoulli_threshold_classification_data();
    let settings = ClassificationSettings::default()
        .with_bernoulli_nb_settings(BernoulliNBParameters::default().with_binarize(0.5));
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_bernoulli_nb();
    let trained = algorithm
        .fit(&x, &y, &settings)
        .expect("Bernoulli NB training should succeed with threshold");
    let predictions = trained
        .predict(&x)
        .expect("Bernoulli NB prediction should succeed with threshold");
    assert_eq!(predictions.len(), y.len());
}

#[test]
fn bernoulli_nb_rejects_non_binary_without_threshold() {
    let (x, y) = bernoulli_threshold_classification_data();
    let params = BernoulliNBParameters::<f64> {
        binarize: None,
        ..BernoulliNBParameters::default()
    };
    let settings = ClassificationSettings::default().with_bernoulli_nb_settings(params);
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_bernoulli_nb();
    let error = algorithm
        .fit(&x, &y, &settings)
        .err()
        .expect("Bernoulli NB training should fail without threshold on non-binary data");
    let message = error.to_string();
    assert!(
        message.contains("Bernoulli naive Bayes requires binary features"),
        "Unexpected error message: {message}"
    );
}

#[test]
fn invalid_alpha_returns_error() {
    // Arrange
    let x = DenseMatrix::from_2d_array(&[
        &[0.0_f32, 0.0_f32],
        &[0.1_f32, 0.0_f32],
        &[0.0_f32, 0.1_f32],
        &[0.1_f32, 0.1_f32],
        &[0.9_f32, 0.9_f32],
        &[1.0_f32, 0.9_f32],
        &[0.9_f32, 1.0_f32],
        &[1.0_f32, 1.0_f32],
        &[0.0_f32, 0.2_f32],
        &[0.2_f32, 0.0_f32],
        &[0.0_f32, 0.3_f32],
        &[0.3_f32, 0.0_f32],
        &[0.8_f32, 0.7_f32],
        &[0.7_f32, 0.8_f32],
        &[0.8_f32, 0.8_f32],
        &[0.7_f32, 0.7_f32],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
    let settings = ClassificationSettings::default().with_logistic_regression_settings(
        LogisticRegressionParameters::default().with_alpha(f64::NAN),
    );
    let algorithm = ClassificationAlgorithm::LogisticRegression(
        smartcore::linear::logistic_regression::LogisticRegression::new(),
    );

    // Act
    let result = algorithm.fit(&x, &y, &settings);

    // Assert
    let message = result.err().unwrap().to_string();
    assert!(
        message.contains("alpha value must be finite"),
        "Unexpected error message: {message}"
    );
}

#[test]
fn support_vector_classifier_requires_settings() {
    let x = DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
    let y: Vec<u32> = vec![0, 1];
    let settings = ClassificationSettings::default();
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_support_vector_classifier();

    let error = algorithm
        .fit(&x, &y, &settings)
        .err()
        .expect("SVC training should fail without settings");
    let message = error.to_string();
    assert!(
        message.contains("support vector classifier settings not provided"),
        "Unexpected error message: {message}"
    );
}

#[test]
fn multinomial_nb_trains_and_predicts_with_non_negative_integers() {
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 0.0_f64, 2.0_f64],
        &[0.0_f64, 1.0_f64, 0.0_f64],
        &[3.0_f64, 0.0_f64, 1.0_f64],
        &[0.0_f64, 0.0_f64, 0.0_f64],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 1, 1];
    let settings = ClassificationSettings::default()
        .with_multinomial_nb_settings(MultinomialNBParameters::default());
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_multinomial_nb();

    let trained = algorithm
        .fit(&x, &y, &settings)
        .expect("training should succeed");
    let predictions = trained.predict(&x).expect("prediction should succeed");
    assert_eq!(predictions.len(), y.len());
}

#[test]
fn multinomial_nb_rejects_fractional_features() {
    let x = DenseMatrix::from_2d_array(&[&[0.5_f64, 1.0_f64], &[1.0_f64, 2.0_f64]]).unwrap();
    let y: Vec<u32> = vec![0, 1];
    let settings = ClassificationSettings::default()
        .with_multinomial_nb_settings(MultinomialNBParameters::default());
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_multinomial_nb();

    let error = algorithm
        .fit(&x, &y, &settings)
        .err()
        .expect("training should fail for fractional counts");
    let message = error.to_string();
    assert!(
        message.contains("requires integer-valued features"),
        "Unexpected error message: {message}"
    );
}

#[test]
fn multinomial_nb_rejects_negative_features() {
    let x = DenseMatrix::from_2d_array(&[&[1.0_f64, -1.0_f64], &[2.0_f64, 0.0_f64]]).unwrap();
    let y: Vec<u32> = vec![0, 1];
    let settings = ClassificationSettings::default()
        .with_multinomial_nb_settings(MultinomialNBParameters::default());
    let algorithm: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
        ClassificationAlgorithm::default_multinomial_nb();

    let error = algorithm
        .fit(&x, &y, &settings)
        .err()
        .expect("training should fail for negative counts");
    let message = error.to_string();
    assert!(
        message.contains("requires non-negative feature values"),
        "Unexpected error message: {message}"
    );
}
