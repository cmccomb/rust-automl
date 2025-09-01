use automl::utils::{CsvError, load_csv_features};
use smartcore::linalg::basic::arrays::Array;

#[test]
fn loads_features_matrix() {
    let x = load_csv_features("tests/fixtures/clustering_points.csv").unwrap();
    assert_eq!(x.shape(), (4, 2));
    assert!((*x.get((0, 0)) - 1.0).abs() < f64::EPSILON);
    assert!((*x.get((3, 1)) - 8.2).abs() < f64::EPSILON);
}

#[test]
fn errors_on_bad_path() {
    let err = load_csv_features("tests/fixtures/does_not_exist.csv").unwrap_err();
    assert!(matches!(err, CsvError::Io(_)));
}

#[test]
fn errors_on_non_numeric() {
    let err = load_csv_features("tests/fixtures/non_numeric_features.csv").unwrap_err();
    assert!(matches!(err, CsvError::Parse(_)));
}

#[test]
fn errors_on_inconsistent_rows() {
    let err = load_csv_features("tests/fixtures/inconsistent_features.csv").unwrap_err();
    assert!(matches!(err, CsvError::Shape(_)));
}
