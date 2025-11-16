#[path = "fixtures/breast_cancer_dataset.rs"]
mod breast_cancer_dataset;
#[path = "fixtures/diabetes_dataset.rs"]
mod diabetes_dataset;

use breast_cancer_dataset::load_breast_cancer_dataset;
use diabetes_dataset::load_diabetes_dataset;
use smartcore::linalg::basic::arrays::Array;

#[test]
fn breast_cancer_dataset_has_expected_shape() {
    let (x, y) = load_breast_cancer_dataset().expect("dataset should load");
    let (rows, cols) = x.shape();
    assert_eq!(rows, 569);
    assert_eq!(cols, 30);
    assert_eq!(y.len(), rows);
    let positives = y.iter().filter(|label| **label == 1).count();
    assert_eq!(positives, 212);
}

#[test]
fn diabetes_dataset_has_expected_shape() {
    let (x, y) = load_diabetes_dataset().expect("dataset should load");
    let (rows, cols) = x.shape();
    assert_eq!(rows, 442);
    assert_eq!(cols, 10);
    assert_eq!(y.len(), rows);
    assert!(y.iter().all(|value| value.is_finite()));
}
