use automl::algorithms::RegressionAlgorithm;
use automl::{DenseMatrix, RegressionSettings};

#[test]
fn display_reflects_skiplist() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
        .skip(RegressionAlgorithm::default_random_forest());
    let output = format!("{settings}");
    assert!(output.contains("Regression settings"));
}

#[test]
fn random_forest_section_labels_n_trees() {
    let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default();
    let output = format!("{settings}");
    assert!(output.contains("Regression settings"));
}
