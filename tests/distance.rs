use automl::settings::KNNParameters;
use automl::utils::distance::{Distance, DistanceError, KNNRegressorDistance};

#[allow(clippy::clone_on_copy)]
fn assert_traits(dist: Distance, display: &str, debug: &str) {
    let copy = dist;
    assert_eq!(copy, dist);
    let cloned = dist.clone();
    assert_eq!(cloned, dist);
    assert_eq!(format!("{dist}"), display);
    assert_eq!(format!("{dist:?}"), debug);
}

#[test]
fn distance_trait_behaviors_and_display() {
    assert_traits(Distance::Euclidean, "Euclidean", "Euclidean");
    assert_traits(Distance::Manhattan, "Manhattan", "Manhattan");
    assert_traits(Distance::Minkowski(4), "Minkowski(p = 4)", "Minkowski(4)");
    assert_traits(Distance::Mahalanobis, "Mahalanobis", "Mahalanobis");
    assert_traits(Distance::Hamming, "Hamming", "Hamming");
}

#[test]
fn knn_distance_from_mahalanobis_errors() {
    let err = KNNRegressorDistance::<f64>::from(Distance::Mahalanobis).unwrap_err();
    assert_eq!(
        err,
        DistanceError::UnsupportedDistance(Distance::Mahalanobis)
    );
}

#[test]
fn knn_param_conversion_mahalanobis_errors() {
    let params = KNNParameters::default().with_distance(Distance::Mahalanobis);
    assert!(matches!(
        params.to_classifier_params::<f64>(),
        Err(DistanceError::UnsupportedDistance(Distance::Mahalanobis))
    ));
    assert!(matches!(
        params.to_regressor_params::<f64>(),
        Err(DistanceError::UnsupportedDistance(Distance::Mahalanobis))
    ));
}
