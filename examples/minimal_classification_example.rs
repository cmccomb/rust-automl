fn main() {
    // Define a default classifier. This include settings, but no data yet.
    let mut classifier = automl::classification::Classifier::default();

    // Load a dataset from smartcore and add it to the classifier
    classifier.with_dataset(smartcore::dataset::breast_cancer::load_dataset());

    // Run a model comparison with all models at default settings
    classifier.compare_models();
}
