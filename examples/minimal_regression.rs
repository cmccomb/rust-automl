use automl::{Settings, SupervisedModel, regression_testing_data};

fn main() {
    // Load some regression data
    let (x, y) = regression_testing_data();

    // Totally customize settings
    let settings = Settings::default_regression();

    // Load a dataset from smartcore and add it to the regressor along with the customized settings
    let mut model = SupervisedModel::new(x, y, settings);

    // Run a model comparison with all models at default settings
    model.train();
}
