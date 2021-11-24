fn main() {
    // Instantiate a regressor
    let mut x = automl::regression::Regressor::default();

    // Add a dataset
    x.with_dataset(smartcore::dataset::diabetes::load_dataset());

    // Compare models and train a final model
    x.auto();

    // Run the demo
    x.run_demo();
}
