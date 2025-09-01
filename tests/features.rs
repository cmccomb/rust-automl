#![allow(clippy::float_cmp)]

use automl::DenseMatrix;
use automl::utils::{FeatureError, interaction_features, polynomial_features};
use smartcore::linalg::basic::arrays::{Array, Array2};

#[test]
fn interaction_adds_pairwise_products() {
    let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let augmented = interaction_features(x.clone()).unwrap();
    assert_eq!(augmented.shape(), (2, 3));
    assert_eq!(*augmented.get((0, 2)), 2.0);
    assert_eq!(*augmented.get((1, 2)), 12.0);
}

#[test]
fn polynomial_adds_squared_and_cross_terms() {
    let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let augmented = polynomial_features(x.clone(), 2).unwrap();
    assert_eq!(augmented.shape(), (2, 5));
    assert_eq!(*augmented.get((0, 2)), 1.0);
    assert_eq!(*augmented.get((0, 3)), 2.0);
    assert_eq!(*augmented.get((0, 4)), 4.0);
}

#[test]
fn interaction_errors_on_empty_input() {
    let x = DenseMatrix::<f64>::zeros(0, 2);
    let err = interaction_features(x).unwrap_err();
    assert_eq!(err, FeatureError::InvalidInputDimensions);
}

#[test]
fn polynomial_errors_on_empty_input() {
    let x = DenseMatrix::<f64>::zeros(0, 2);
    let err = polynomial_features(x, 2).unwrap_err();
    assert_eq!(err, FeatureError::InvalidInputDimensions);
}
