#[path = "fixtures/classification_data.rs"]
mod classification_data;

use automl::settings::ClassificationSettings;
use automl::{ClassificationModel, DenseMatrix};
use classification_data::classification_testing_data;

#[test]
fn test_default_classification() {
    let settings = ClassificationSettings::default();
    test_from_settings(settings);
}

fn test_from_settings(settings: ClassificationSettings) {
    let (x, y) = classification_testing_data();

    let mut model = ClassificationModel::new(x, y, settings);
    model.train();

    model.predict(DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap());
}
