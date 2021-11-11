fn main() {
    let data = linfa_datasets::diabetes();
    let r = automl::regression::compare_models(&data);
    print!("{}", r);
}
