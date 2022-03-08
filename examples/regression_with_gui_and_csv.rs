use automl::settings::FinalModel;

fn main() {
    #[cfg(feature = "csv")]
    {
        // Define a default regressor from a dataset
        let file_name = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

        // Set up the regressor settings and load data
        let settings = automl::Settings::default_regression().with_number_of_folds(2);

        let mut model = automl::SupervisedModel::new_from_csv(file_name, 8, true, settings);

        // Run a model comparison and train a final model
        model.train();

        // Run a graphical demo of the model if the `gui` feature is enabled
        #[cfg(feature = "gui")]
        model.run_gui();

        // Panic if the `gui` feature is not enabled
        #[cfg(not(feature = "gui"))]
        panic!("You must enable the `gui` feature for this example to work correctly.")
    }
}
