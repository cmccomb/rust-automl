#[cfg(test)]
mod regression_tests {
    #[test]
    fn test_with_default_settings() {
        use smartcore::linalg::naive::dense_matrix::DenseMatrix;
        use smartcore::model_selection::train_test_split;
        // Check training
        let data = smartcore::dataset::diabetes::load_dataset();
        let settings = automl::regression::Settings::default();
        let results = automl::regression::compare_models(data, settings);

        print!("{}", results);

        // Check inference
        let dataset = smartcore::dataset::diabetes::load_dataset();
        let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        // These are our target values
        let y = dataset.target;

        let (_, x, _, _) = train_test_split(&x, &y, 0.5, true);

        print!("{:?}", results.predict_with_best_model(&x));
    }
}
