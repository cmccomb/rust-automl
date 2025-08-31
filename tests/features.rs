#![allow(clippy::float_cmp)]

use automl::DenseMatrix;
use automl::utils::{interaction_features, polynomial_features};
use smartcore::linalg::basic::arrays::Array;

#[test]
fn interaction_adds_pairwise_products() {
    let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let augmented = interaction_features(x.clone());
    assert_eq!(augmented.shape(), (2, 3));
    assert_eq!(*augmented.get((0, 2)), 2.0);
    assert_eq!(*augmented.get((1, 2)), 12.0);
}

#[test]
fn polynomial_adds_squared_and_cross_terms() {
    let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let augmented = polynomial_features(x.clone(), 2);
    assert_eq!(augmented.shape(), (2, 5));
    assert_eq!(*augmented.get((0, 2)), 1.0);
    assert_eq!(*augmented.get((0, 3)), 2.0);
    assert_eq!(*augmented.get((0, 4)), 4.0);
}
