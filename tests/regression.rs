#[cfg(test)]
mod regression_tests {
    use automl::{settings::*, *};
    use smartcore::dataset::diabetes::load_dataset;

    #[test]
    fn test_step_by_step() {
        // Check training
        let dataset = load_dataset();

        // Set up the regressor settings and load data
        let settings = Settings::default_regression()
            .sorted_by(Metric::MeanAbsoluteError)
            .with_svr_settings(SVRParameters::default().with_eps(2.0).with_c(10.0))
            .skip(Algorithm::ElasticNet)
            .with_number_of_folds(2)
            .with_final_model(FinalModel::Blend)
            .with_validation_fraction(0.3);
        let mut regressor = SupervisedModel::new_from_dataset(dataset, settings);

        // Compare models
        regressor.train();

        // Do inference with final model
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; 1];
        print!("{:?}", regressor.predict_from_vec(x));
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_read_from_csv() {
        let file_name = "data/diabetes.csv";

        // Set up the regressor settings and load data
        let settings = Settings::default_regression()
            .sorted_by(Metric::MeanSquaredError)
            .with_svr_settings(SVRParameters::default().with_eps(2.0).with_c(10.0))
            .skip(Algorithm::ElasticNet)
            .with_number_of_folds(2);

        let mut regressor = SupervisedModel::new_from_csv(file_name, 10, true, settings);

        regressor.train();

        // Do inference with final model
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; 1];
        print!("{:?}", regressor.predict_from_vec(x));
    }
}
