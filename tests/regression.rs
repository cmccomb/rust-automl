#[cfg(test)]
mod regression_tests {
    #[test]
    fn test_with_default_settings() {
        use automl::regression::{compare_models, Metric, Regressor, Settings};
        use smartcore::svm::svr::SVRParameters;
        use smartcore::{
            dataset::diabetes::load_dataset, linalg::naive::dense_matrix::DenseMatrix,
            model_selection::train_test_split,
        };

        // Check training
        let data = load_dataset();
        let settings = Settings::default()
            .sorted_by(automl::regression::Metric::MeanSquaredError)
            .with_svr_settings(SVRParameters::default().with_eps(2.0).with_c(10.0))
            .skip(vec![Regressor::ElasticNet]);
        let results = compare_models(data, settings);

        print!("{}", results);

        // Check inference
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]);

        print!("{:?}", results.predict_with_best_model(&x));
    }
}
