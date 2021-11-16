#[cfg(test)]
mod classification_tests {
    #[test]
    fn test_step_by_step() {
        use automl::classification::{Classifier, Settings};
        use smartcore::dataset::breast_cancer::load_dataset;

        // Check training
        let dataset = load_dataset();

        // Set up the regressor settings and load data
        let settings = Settings::default();
        let mut regressor = Classifier::new(settings);
        regressor.with_dataset(dataset);

        // Compare models
        regressor.compare_models();
        print!("{}", regressor);

        // Train final model
        regressor.train_final_model();
    }
}
