#[cfg(test)]
mod classification_tests {
    #[test]
    fn print_settings() {
        let settings = automl::classification::Settings::default()
            .skip_algorithms(vec![
                automl::classification::Algorithm::DecisionTree,
                automl::classification::Algorithm::LogisticRegression,
            ])
            .with_number_of_folds(3);
        println!("{}", &settings);
    }

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
