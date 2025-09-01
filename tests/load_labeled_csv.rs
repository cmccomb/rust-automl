use automl::utils::load_labeled_csv;
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
