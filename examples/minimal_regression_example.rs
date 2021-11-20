fn main() {
    // Define a default regressor. This include settings, but no data yet.
    let mut regressor = automl::regression::Regressor::default();

    // Load a dataset from smartcore and add it to the regressor
    regressor.with_dataset(smartcore::dataset::diabetes::load_dataset());

    // Run a model comparison with all models at default settings
    regressor.compare_models();
}
