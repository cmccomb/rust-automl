use automl::settings::FinalModel;

fn main() {
    // Define a default regressor from a dataset
    let mut model = automl::SupervisedModel::new_from_dataset(
        smartcore::dataset::diabetes::load_dataset(),
        automl::Settings::default_regression().with_final_model(FinalModel::default_blending()),
    );

    // Run a model comparison and train a final model
    model.train();

    // Run a graphical demo of the model if the `gui` feature is enabled
    #[cfg(feature = "gui")]
    model.run_gui();

    // Panic if the `gui` feature is not enabled
    #[cfg(not(feature = "gui"))]
    panic!("You must enable the `gui` feature for this example to work correctly.")
}
