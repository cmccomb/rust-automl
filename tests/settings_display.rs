use automl::{DenseMatrix, Settings, settings::Algorithm};

#[test]
fn display_reflects_skiplist() {
    let settings = Settings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default_regression()
        .skip(Algorithm::default_random_forest());
    let output = format!("{settings}");
    assert!(output.contains("Linear Regressor"));
    assert_eq!(output.matches("Random Forest Regressor").count(), 1);
}

#[test]
fn random_forest_section_labels_n_trees() {
    let settings = Settings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default_regression();
    let output = format!("{settings}");
    assert!(output.contains("Number of Trees"));
}
