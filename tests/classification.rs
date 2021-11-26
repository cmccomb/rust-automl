#[cfg(test)]
mod classification_tests {
    use automl::*;
    #[test]
    fn test_step_by_step() {
        // Check training
        let dataset = smartcore::dataset::breast_cancer::load_dataset();

        // Set up the regressor settings and load data
        let settings = Settings::default_classification().with_number_of_folds(2);
        let mut regressor = SupervisedModel::new_from_dataset(dataset, settings);

        // Compare models
        regressor.compare_models();

        // Train final model
        regressor.train_final_model();
    }
}
