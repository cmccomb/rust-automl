#[cfg(test)]
mod new_from_dataset {
    // use smartcore::dataset::breast_cancer;
    // use smartcore::dataset::diabetes;

    // #[test]
    // fn classification() {
    //     // Make a model
    //     let mut classifier = SupervisedModel::new(
    //         breast_cancer::load_dataset(),
    //         Settings::default_classification(),
    //     );
    //
    //     // Compare models
    //     classifier.train();
    //
    //     // Try to predict something from a vector
    //     classifier.predict(vec![vec![5.0_f32; 30]; 10]);
    //
    //     // Try to predict something from ndarray
    //     #[cfg(feature = "nd")]
    //     classifier.predict(ndarray::Array2::from_shape_vec((10, 30), vec![5.0; 300]).unwrap());
    //
    //     // Try to predict something from a csv
    //     #[cfg(feature = "csv")]
    //     classifier.predict("data/breast_cancer_without_target.csv");
    // }

    // #[test]
    // fn regression() {
    //     // Make a model
    //     let mut regressor =
    //         SupervisedModel::new(diabetes::load_dataset(), Settings::default_regression());
    //
    //     // Compare models
    //     regressor.train();
    //
    //     // Try to predict something from a vector
    //     regressor.predict(vec![vec![5.0_f32; 10]; 10]);
    //
    //     // Try to predict something from ndarray
    //     #[cfg(feature = "nd")]
    //     regressor.predict(ndarray::Array2::from_shape_vec((10, 10), vec![5.0; 100]).unwrap());
    //
    //     // Try to predict something from a csv
    //     #[cfg(feature = "csv")]
    //     regressor.predict("data/diabetes_without_target.csv");
    // }
}
