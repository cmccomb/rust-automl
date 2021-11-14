#[cfg(test)]
mod regression_tests {
    #[test]
    fn test_step_by_step() {
        use automl::regression::{Algorithm, Metric, Regressor, Settings};
        use smartcore::svm::svr::SVRParameters;
        use smartcore::{
            dataset::diabetes::load_dataset, linalg::naive::dense_matrix::DenseMatrix,
            model_selection::train_test_split,
        };

        // Check training
        let dataset = load_dataset();
        let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        let y = dataset.target;

        // Set up the regressor settings and load data
        let settings = Settings::default()
            .sorted_by(Metric::MeanSquaredError)
            .with_svr_settings(SVRParameters::default().with_eps(2.0).with_c(10.0))
            .skip(vec![Algorithm::ElasticNet]);
        let mut regressor = Regressor::new(settings);
        regressor.with_data(x, y);

        // Compare models
        regressor.compare_models();
        print!("{}", regressor);

        // Train final model
        regressor.train_final_model();

        // Do inference with final model
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]);
        print!("{:?}", regressor.predict(&x));
    }

    #[test]
    fn test_auto() {
        use automl::regression::{Algorithm, Metric, Regressor, Settings};
        use smartcore::svm::svr::SVRParameters;
        use smartcore::{
            dataset::diabetes::load_dataset, linalg::naive::dense_matrix::DenseMatrix,
            model_selection::train_test_split,
        };

        // Check training
        let dataset = load_dataset();
        let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        let y = dataset.target;

        // Set up the regressor settings and load data
        let settings = Settings::default()
            .sorted_by(Metric::MeanSquaredError)
            .with_svr_settings(SVRParameters::default().with_eps(2.0).with_c(10.0))
            .skip(vec![Algorithm::ElasticNet]);

        // Compare models
        let regressor = Regressor::auto(settings, x, y);

        // Do inference with final model
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]);
        print!("{:?}", regressor.predict(&x));
    }
}
