#[cfg(test)]
mod new_from_dataset {
    use automl::{settings::*, *};
    use smartcore::dataset::breast_cancer;
    use smartcore::dataset::diabetes;

    #[test]
    fn classification() {
        // Make a model
        let mut classifier = SupervisedModel::new_from_dataset(
            breast_cancer::load_dataset(),
            Settings::default_classification(),
        );

        // Compare models
        classifier.train();

        // Try to predict something from a vector
        classifier.predict_from_vec(vec![vec![5.0 as f32; 30]; 10]);

        // Try to predict something from ndarray
        #[cfg(feature = "nd")]
        classifier.predict_from_ndarray(
            ndarray::Array2::from_shape_vec((10, 30), vec![5.0; 300]).unwrap(),
        );

        // Try to predict something from a csv
        #[cfg(feature = "csv")]
        classifier.predict_from_csv("data/breast_cancer_without_target.csv", true);
    }

    #[test]
    fn regression() {
        // Make a model
        let mut regressor = SupervisedModel::new_from_dataset(
            diabetes::load_dataset(),
            Settings::default_regression(),
        );

        // Compare models
        regressor.train();

        // Try to predict something from a vector
        regressor.predict_from_vec(vec![vec![5.0 as f32; 10]; 10]);

        // Try to predict something from ndarray
        #[cfg(feature = "nd")]
        regressor.predict_from_ndarray(
            ndarray::Array2::from_shape_vec((10, 10), vec![5.0; 100]).unwrap(),
        );

        // Try to predict something from a csv
        #[cfg(feature = "csv")]
        regressor.predict_from_csv("data/diabetes_without_target.csv", true);
    }
}
