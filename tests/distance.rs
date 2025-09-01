use automl::utils::Distance;

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
