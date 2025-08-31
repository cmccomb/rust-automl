use automl::DenseMatrix;
use automl::settings::Algorithm;

type Alg = Algorithm<f64, f64, DenseMatrix<f64>, Vec<f64>>;

#[test]
fn default_equals_linear() {
    assert!(Alg::default() == Alg::default_linear());
}

#[test]
fn all_algorithms_contains_linear() {
    let algorithms = Alg::all_algorithms();
    assert!(algorithms.iter().any(|a| matches!(a, Alg::Linear(_))));
}
