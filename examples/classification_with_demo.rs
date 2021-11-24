fn main() {
    // Instantiate a classifier
    let mut x = automl::classification::Classifier::default();

    // Add a dataset
    x.with_dataset(smartcore::dataset::breast_cancer::load_dataset());

    // Compare models and train a final one
    x.auto();

    // Run teh demo
    x.run_demo();
}
