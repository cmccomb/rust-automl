fn main() {
    #[cfg(feature = "gui")]
    {
        // Define a default regressor from a dataset
        let mut model = automl::SupervisedModel::new_from_dataset(
            smartcore::dataset::diabetes::load_dataset(),
            automl::Settings::default_regression(),
        );

        // Run a model comparison and train a final model
        model.auto();

        // Run a graphical demo of the model
        model.run_gui();
    }
}
