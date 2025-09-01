#[path = "fixtures/classification_data.rs"]
mod classification_data;

use automl::algorithms::ClassificationAlgorithm;
use automl::settings::{
    ClassificationSettings, KNNAlgorithmName, KNNClassifierParameters, KNNWeightFunction,
    RandomForestClassifierParameters, build_knn_classifier_parameters,
};
use automl::{ClassificationModel, DenseMatrix};
use classification_data::classification_testing_data;
use smartcore::metrics::distance::euclidian::Euclidian;

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
    let algorithms =
        ClassificationAlgorithm::<f64, i32, DenseMatrix<f64>, Vec<i32>>::all_algorithms(&settings);
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
}

#[test]
fn test_build_knn_classifier_parameters_helper() {
    let settings = KNNClassifierParameters::default()
        .with_k(4)
        .with_algorithm(KNNAlgorithmName::LinearSearch)
        .with_weight(KNNWeightFunction::Distance);
    let params = build_knn_classifier_parameters::<f64, _>(&settings, Euclidian::new());
    assert_eq!(params.k, 4);
    assert!(matches!(params.algorithm, KNNAlgorithmName::LinearSearch));
    assert!(matches!(params.weight, KNNWeightFunction::Distance));
}

fn test_from_settings(settings: ClassificationSettings) {
    let (x, y) = classification_testing_data();

    let mut model = ClassificationModel::new(x, y, settings);
    model.train();

    model.predict(DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap());
}
