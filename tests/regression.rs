#[cfg(test)]
mod regression_tests {
    use automl::*;
    use smartcore::linalg::basic::matrix::DenseMatrix;
    use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
    use smartcore::linalg::traits::qr::QRDecomposable;
    use smartcore::linalg::traits::svd::SVDDecomposable;

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
        regressor.predict(vec![vec![5.0_f32; 10]; 10]);
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
        regressor.predict(vec![vec![5.0_f32; 8]; 8]);
    }

    // #[test]
    // fn test_add_interactions_preprocessing() {
    //     let settings =
    //         Settings::default_regression().with_preprocessing(PreProcessing::AddInteractions);
    //     test_from_settings(settings);
    // }
    //
    // #[test]
    // fn test_add_polynomial_preprocessing() {
    //     let settings = Settings::default_regression()
    //         .with_preprocessing(PreProcessing::AddPolynomial { order: 2 });
    //     test_from_settings(settings);
    // }
    //
    // #[test]
    // fn test_blending() {
    //     let settings = Settings::default_regression().with_final_model(FinalAlgorithm::Blending {
    //         algorithm: Algorithm::Linear,
    //         meta_training_fraction: 0.15,
    //         meta_testing_fraction: 0.15,
    //     });
    //     test_from_settings(settings);
    // }

    fn test_from_settings(settings: Settings<f64, f64, DenseMatrix<f64>, Vec<f64>>) {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ])
        .unwrap();

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];
        // Check training

        // Set up the regressor settings and load data
        let mut regressor = SupervisedModel::build(x, y, settings);

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
