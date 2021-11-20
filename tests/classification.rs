#[cfg(test)]
mod classification_tests {
    #[test]
    fn test_step_by_step() {
        // Check training
        let dataset = smartcore::dataset::breast_cancer::load_dataset();

        // Set up the regressor settings and load data
        let settings = automl::classification::Settings::default().with_number_of_folds(2);
        let mut regressor = automl::classification::Classifier::default();
        regressor.with_settings(settings);
        regressor.with_dataset(dataset);

        // Compare models
        regressor.compare_models();
        print!("{}", regressor);

        // Train final model
        regressor.train_final_model();
    }
}
