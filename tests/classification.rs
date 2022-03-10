#[cfg(test)]
mod classification_tests {
    use automl::{settings::*, *};
    use smartcore::dataset::breast_cancer::load_dataset;

    #[test]
    #[cfg(feature = "csv")]
    fn test_new_from_csv() {
        let file_name = "data/breast_cancer.csv";

        // Set up the classifier settings and load data
        let settings = Settings::default_classification().with_number_of_folds(2);

        let mut classifier = SupervisedModel::new((file_name, 30, true), settings);

        // Compare models
        classifier.train();

        // Try to predict something
        classifier.predict(vec![vec![5.0 as f32; 30]; 10]);
        classifier.predict(("data/breast_cancer_without_target.csv", true));
        #[cfg(feature = "nd")]
        classifier.predict(ndarray::Array2::from_shape_vec((10, 30), vec![5.0; 300]).unwrap());
    }

    #[test]
    fn test_add_interactions_preprocessing() {
        let settings =
            Settings::default_classification().with_preprocessing(PreProcessing::AddInteractions);
        test_from_settings(settings);
    }

    #[test]
    fn test_add_polynomial_preprocessing() {
        let settings = Settings::default_classification()
            .with_preprocessing(PreProcessing::AddPolynomial { order: 2 });
        test_from_settings(settings);
    }

    #[test]
    fn test_blending() {
        let settings = Settings::default_classification().with_final_model(FinalModel::Blending {
            algorithm: Algorithm::LogisticRegression,
            meta_training_fraction: 0.15,
            meta_testing_fraction: 0.15,
        });
        test_from_settings(settings);
    }

    fn test_from_settings(settings: Settings) {
        // Check training
        let dataset = load_dataset();

        // Set up the regressor settings and load data
        let mut classifier = SupervisedModel::new(dataset, settings);

        // Compare models
        classifier.train();

        // Try to predict something
        classifier.predict(vec![vec![5.0 as f32; 30]; 10]);
    }
}
