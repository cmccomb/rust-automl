use automl::algorithms::RegressionAlgorithm;
use automl::settings::{Distance, KNNAlgorithmName, KNNParameters, KNNWeightFunction};
use automl::utils::distance::KNNRegressorDistance;
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

#[test]
fn knn_param_conversion_matches_for_classifier_and_regressor() {
    for distance in [
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::Minkowski(3),
        Distance::Hamming,
    ] {
        let params = KNNParameters::default()
            .with_k(7)
            .with_weight(KNNWeightFunction::Distance)
            .with_algorithm(KNNAlgorithmName::LinearSearch)
            .with_distance(distance);
        let classifier = params.to_classifier_params::<f64>().unwrap();
        let regressor = params.to_regressor_params::<f64>().unwrap();

        assert_eq!(classifier.k, regressor.k);
        assert_eq!(
            format!("{:?}", classifier.weight),
            format!("{:?}", regressor.weight)
        );
        assert_eq!(
            format!("{:?}", classifier.algorithm),
            format!("{:?}", regressor.algorithm)
        );

        match distance {
            Distance::Euclidean => {
                assert!(matches!(
                    classifier.distance,
                    KNNRegressorDistance::Euclidean(_)
                ));
                assert!(format!("{regressor:?}").contains("Euclidian"));
            }
            Distance::Manhattan => {
                assert!(matches!(
                    classifier.distance,
                    KNNRegressorDistance::Manhattan(_)
                ));
                assert!(format!("{regressor:?}").contains("Manhattan"));
            }
            Distance::Minkowski(_) => {
                assert!(matches!(
                    classifier.distance,
                    KNNRegressorDistance::Minkowski(_)
                ));
                assert!(format!("{regressor:?}").contains("Minkowski"));
            }
            Distance::Hamming => {
                assert!(matches!(
                    classifier.distance,
                    KNNRegressorDistance::Hamming(_)
                ));
                assert!(format!("{regressor:?}").contains("Hamming"));
            }
            Distance::Mahalanobis => unreachable!(),
        }
    }
}
