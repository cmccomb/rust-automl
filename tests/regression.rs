#[cfg(test)]
mod regression_tests {
    #[test]
    fn test_with_default_settings() {
        let data = smartcore::dataset::diabetes::load_dataset();
        let settings = automl::regression::Settings::default();
        let x = automl::regression::compare_models(data, settings);
        print!("{}", x);
    }
}
