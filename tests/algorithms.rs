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
fn all_algorithms_includes_knn_for_supported_distances() {
    for distance in [
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::Minkowski(3),
        Distance::Hamming,
    ] {
        let settings = RegressionSettings::default()
            .with_knn_regressor_settings(KNNParameters::default().with_distance(distance));
        let algorithms = Alg::all_algorithms(&settings);
        assert!(algorithms.iter().any(|a| matches!(a, Alg::KNNRegressor(_))));
    }
}

#[test]
fn all_algorithms_skips_knn_for_mahalanobis() {
    let settings = RegressionSettings::default()
        .with_knn_regressor_settings(KNNParameters::default().with_distance(Distance::Mahalanobis));
    let algorithms = Alg::all_algorithms(&settings);
    assert!(
        algorithms
            .iter()
            .all(|a| !matches!(a, Alg::KNNRegressor(_)))
    );
}
