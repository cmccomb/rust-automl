fn main() {
    // // Set up and train a classification model with only one algorithm for simplicity
    // let settings = Settings::default_classification().only(Algorithm::LogisticRegression);
    // let dataset = smartcore::dataset::breast_cancer::load_dataset();
    // let mut model = SupervisedModel::new(dataset, settings);
    // model.train();
    //
    // // Save the best model
    // let file_name = "examples/best_model_only.sc";
    // model.save_best(file_name);
    //
    // // Load that model for use directly in SmartCore
    // let mut buf: Vec<u8> = Vec::new();
    // std::fs::File::open(file_name)
    //     .and_then(|mut f| f.read_to_end(&mut buf))
    //     .expect("Cannot load model from file.");
    // let model: LogisticRegression<f32, DenseMatrix<f32>> =
    //     bincode::deserialize(&buf).expect("Can not deserialize the model");
    //
    // // Use the model variable to prove that this works.
    // println!("{:?}", model.coefficients());
    // println!("{:?}", model.intercept());
}
