#[cfg(test)]
mod classification_tests {
    #[test]
    fn test_with_default_settings() {
        let data = smartcore::dataset::breast_cancer::load_dataset();
        let settings = automl::classification::Settings::default();
        let x = automl::classification::compare_models(data, settings);
        print!("{}", x);
    }
}
