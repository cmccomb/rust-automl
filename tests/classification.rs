#[cfg(test)]
mod classification_tests {
    use automl::{settings::*, *};
    use smartcore::dataset::breast_cancer::load_dataset;

    #[test]
    fn test_new_from_dataset() {
        // Check training
        let dataset = load_dataset();

        // Set up the classifier settings and load data
        let settings = Settings::default_classification().with_number_of_folds(2);
        let mut classifier = SupervisedModel::new_from_dataset(dataset, settings);

        // Compare models
        classifier.train();

        // Try to predict something
        classifier.predict_from_vec(vec![vec![5.0 as f32; 30]; 10]);
        #[cfg(feature = "nd")]
        classifier.predict_from_ndarray(
            ndarray::Array2::from_shape_vec((10, 30), vec![5.0; 300]).unwrap(),
        );
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_new_from_csv() {
        let file_name = "data/breast_cancer.csv";

        // Set up the classifier settings and load data
        let settings = Settings::default_classification().with_number_of_folds(2);

        let mut classifier = SupervisedModel::new_from_csv(file_name, 9, true, settings);

        // Compare models
        classifier.train();

        // Try to predict something
        classifier.predict_from_vec(vec![vec![5.0 as f32; 9]; 10]);
        #[cfg(feature = "nd")]
        classifier
            .predict_from_ndarray(ndarray::Array2::from_shape_vec((10, 9), vec![5.0; 90]).unwrap());
    }
}
