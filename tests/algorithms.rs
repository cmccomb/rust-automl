use automl::algorithms::RegressionAlgorithm;
use automl::settings::{Distance, KNNParameters};
use automl::{DenseMatrix, RegressionSettings};

type Alg = RegressionAlgorithm<f64, f64, DenseMatrix<f64>, Vec<f64>>;

#[test]
fn default_equals_linear() {
    assert!(Alg::default() == Alg::default_linear());
}

#[test]
fn all_algorithms_contains_linear() {
    let algorithms = Alg::all_algorithms(&RegressionSettings::default());
    assert!(algorithms.iter().any(|a| matches!(a, Alg::Linear(_))));
}

#[test]
fn all_algorithms_respects_knn_distance() {
    let settings = RegressionSettings::default()
        .with_knn_regressor_settings(KNNParameters::default().with_distance(Distance::Manhattan));
    let algorithms = Alg::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .any(|a| matches!(a, Alg::KNNRegressorManhattan(_)))
    );
    assert!(
        algorithms
            .iter()
            .all(|a| !matches!(a, Alg::KNNRegressorEuclidian(_)))
    );
}
