#[cfg(test)]
mod regression_tests {
    use automl::{settings::*, *};
    use smartcore::dataset::diabetes::load_dataset;

    #[test]
    #[cfg(feature = "csv")]
    fn test_new_from_csv() {
        let file_name = "data/diabetes.csv";

        // Set up the regressor settings and load data
        let settings = Settings::default_regression().with_number_of_folds(2);

        let mut regressor = SupervisedModel::new((file_name, 10), settings);

        // Compare models
        regressor.train();

        // Try to predict something
        regressor.predict(vec![vec![5.0 as f32; 10]; 10]);
        regressor.predict("data/diabetes_without_target.csv");
        #[cfg(feature = "nd")]
        regressor.predict(ndarray::Array2::from_shape_vec((10, 10), vec![5.0; 100]).unwrap());
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_new_from_csv_url() {
        // let file_name = "data/diabetes.csv";
        let file_name = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

        // Set up the regressor settings and load data
        let settings = Settings::default_regression().with_number_of_folds(2);

        let mut regressor = SupervisedModel::new((file_name, 8), settings);

        // Compare models
        regressor.train();

        // Try to predict something
        regressor.predict(vec![vec![5.0 as f32; 8]; 8]);
    }

    #[test]
    fn test_add_interactions_preprocessing() {
        let settings =
            Settings::default_regression().with_preprocessing(PreProcessing::AddInteractions);
        test_from_settings(settings);
    }

    #[test]
    fn test_add_polynomial_preprocessing() {
        let settings = Settings::default_regression()
            .with_preprocessing(PreProcessing::AddPolynomial { order: 2 });
        test_from_settings(settings);
    }

    #[test]
    fn test_blending() {
        let settings = Settings::default_regression().with_final_model(FinalModel::Blending {
            algorithm: Algorithm::Linear,
            meta_training_fraction: 0.15,
            meta_testing_fraction: 0.15,
        });
        test_from_settings(settings);
    }

    fn test_from_settings(settings: Settings) {
        // Check training
        let dataset = load_dataset();

        // Set up the regressor settings and load data
        let mut regressor = SupervisedModel::new(dataset, settings);

        // Compare models
        regressor.train();

        // Try to predict something
        regressor.predict(vec![vec![5.0_f32; 10]; 10]);
    }
}
