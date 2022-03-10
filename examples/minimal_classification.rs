fn main() {
    // Define a default regressor from a dataset
    let mut model = automl::SupervisedModel::new(
        smartcore::dataset::breast_cancer::load_dataset(),
        automl::Settings::default_classification(),
    );

    // Run a model comparison with all models at default settings
    model.train();
}
