#[cfg(test)]
mod regression_tests {
    use automl::{
        DenseMatrix, regression_testing_data, settings::Settings, supervised_model::SupervisedModel,
    };

    #[test]
    fn test_default_regression() {
        let settings = Settings::default_regression();
        test_from_settings(settings);
    }

    fn test_from_settings(settings: Settings<f64, f64, DenseMatrix<f64>, Vec<f64>>) {
        // Get test data
        let (x, y) = regression_testing_data();

        // Set up the regressor settings and load data
        let mut regressor = SupervisedModel::new(x, y, settings);

        // Compare models
        regressor.train();

        // Try to predict something
        regressor.predict(
            smartcore::linalg::basic::matrix::DenseMatrix::from_2d_array(&[
                &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
                &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            ])
            .unwrap(),
        );
    }
}
