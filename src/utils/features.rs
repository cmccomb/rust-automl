//! Feature engineering helpers.
//!
//! This module provides pure functions to expand feature sets, keeping
//! transformations separate from model logic.

use itertools::Itertools;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

use crate::utils::math::elementwise_multiply;

/// Generate pairwise interaction features by multiplying each combination of
/// columns.
///
/// # Arguments
///
/// * `x` - Input feature matrix.
///
/// # Examples
///
/// ```
/// use automl::utils::interaction_features;
/// use smartcore::linalg::basic::arrays::Array;
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
/// let augmented = interaction_features(x);
/// assert_eq!(augmented.shape(), (2, 3));
/// ```
///
/// # Panics
///
/// Panics if a new feature column cannot be added to the matrix.
pub fn interaction_features<INPUT, InputArray>(mut x: InputArray) -> InputArray
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone + Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (height, width) = x.shape();
    for column_1 in 0..width {
        for column_2 in (column_1 + 1)..width {
            let col1: Vec<INPUT> = (0..height)
                .map(|idx| *x.get_col(column_1).get(idx))
                .collect();
            let col2: Vec<INPUT> = (0..height)
                .map(|idx| *x.get_col(column_2).get(idx))
                .collect();
            let feature = elementwise_multiply(&col1, &col2);
            let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                .expect("Cannot create matrix")
                .transpose();
            x = x.h_stack(&new_column);
        }
    }
    x
}

/// Generate polynomial features up to a given order.
///
/// # Arguments
///
/// * `x` - Input feature matrix.
/// * `order` - Highest polynomial order to generate.
///
/// # Examples
///
/// ```
/// use automl::utils::polynomial_features;
/// use smartcore::linalg::basic::arrays::Array;
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
/// let augmented = polynomial_features(x, 2);
/// assert_eq!(augmented.shape(), (2, 5));
/// ```
///
/// # Panics
///
/// Panics if a new feature column cannot be added to the matrix.
pub fn polynomial_features<INPUT, InputArray>(mut x: InputArray, order: usize) -> InputArray
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone + Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (height, width) = x.shape();
    for n in 2..=order {
        let combinations = (0..width).combinations_with_replacement(n);
        for combo in combinations {
            let mut feature: Vec<INPUT> = vec![INPUT::one(); height];
            for column in combo {
                let col: Vec<INPUT> = (0..height).map(|idx| *x.get_col(column).get(idx)).collect();
                feature = elementwise_multiply(&col, &feature);
            }
            let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                .expect("Cannot create matrix")
                .transpose();
            x = x.h_stack(&new_column);
        }
    }
    x
}
