#[path = "fixtures/classification_data.rs"]
mod classification_data;

use automl::algorithms::ClassificationAlgorithm;
use automl::settings::{ClassificationSettings, RandomForestClassifierParameters};
use automl::{ClassificationModel, DenseMatrix};
use classification_data::classification_testing_data;

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
}

fn test_from_settings(settings: ClassificationSettings) {
    let (x, y) = classification_testing_data();

    let mut model = ClassificationModel::new(x, y, settings);
    model.train();

    model.predict(DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap());
}
