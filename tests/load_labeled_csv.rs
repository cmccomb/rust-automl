use automl::utils::{CsvError, load_labeled_csv};
use smartcore::linalg::basic::arrays::Array;

#[test]
fn separates_features_and_target() {
    let (x, y) = load_labeled_csv("tests/fixtures/supervised_sample.csv", 2).unwrap();
    assert_eq!(x.shape(), (3, 2));
    let expected_y = [3.0, 6.0, 9.0];
    for (actual, expected) in y.iter().zip(expected_y) {
        assert!((*actual - expected).abs() < f64::EPSILON);
    }
    assert!((*x.get((0, 0)) - 1.0).abs() < f64::EPSILON);
    assert!((*x.get((2, 1)) - 8.0).abs() < f64::EPSILON);
}

#[test]
fn errors_on_bad_path() {
    let err = load_labeled_csv("tests/fixtures/does_not_exist.csv", 0).unwrap_err();
    assert!(matches!(err, CsvError::Io(_)));
}

#[test]
fn errors_on_non_numeric() {
    let err = load_labeled_csv("tests/fixtures/non_numeric_labeled.csv", 2).unwrap_err();
    assert!(matches!(err, CsvError::Parse(_)));
}

#[test]
fn errors_on_inconsistent_rows() {
    let err = load_labeled_csv("tests/fixtures/inconsistent_labeled.csv", 2).unwrap_err();
    assert!(matches!(err, CsvError::Shape(_)));
}
