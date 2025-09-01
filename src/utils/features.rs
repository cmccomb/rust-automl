//! Feature engineering helpers.
//!
//! This module provides pure functions to expand feature sets, keeping
//! transformations separate from model logic.

use itertools::Itertools;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::utils::math::elementwise_multiply;

/// Errors that can occur during feature construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureError {
    /// Failed to build an intermediate matrix.
    MatrixCreationFailed,
    /// The input matrix has zero height or width.
    InvalidInputDimensions,
}

impl Display for FeatureError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixCreationFailed => write!(f, "cannot create matrix"),
            Self::InvalidInputDimensions => write!(f, "input matrix has invalid dimensions"),
        }
    }
}

impl Error for FeatureError {}

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
/// use automl::utils::FeatureError;
/// # fn main() -> Result<(), FeatureError> {
/// let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
/// let augmented = interaction_features(x)?;
/// assert_eq!(augmented.shape(), (2, 3));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if the input matrix is empty or a new feature column
/// cannot be created.
pub fn interaction_features<INPUT, InputArray>(
    mut x: InputArray,
) -> Result<InputArray, FeatureError>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone + Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (height, width) = x.shape();
    if height == 0 || width == 0 {
        return Err(FeatureError::InvalidInputDimensions);
    }
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
                .map_err(|_| FeatureError::MatrixCreationFailed)?
                .transpose();
            x = x.h_stack(&new_column);
        }
    }
    Ok(x)
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
/// use automl::utils::FeatureError;
/// # fn main() -> Result<(), FeatureError> {
/// let x = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
/// let augmented = polynomial_features(x, 2)?;
/// assert_eq!(augmented.shape(), (2, 5));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if the input matrix is empty or a new feature column
/// cannot be created.
pub fn polynomial_features<INPUT, InputArray>(
    mut x: InputArray,
    order: usize,
) -> Result<InputArray, FeatureError>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone + Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (height, width) = x.shape();
    if height == 0 || width == 0 {
        return Err(FeatureError::InvalidInputDimensions);
    }
    for n in 2..=order {
        let combinations = (0..width).combinations_with_replacement(n);
        for combo in combinations {
            let mut feature: Vec<INPUT> = vec![INPUT::one(); height];
            for column in combo {
                let col: Vec<INPUT> = (0..height).map(|idx| *x.get_col(column).get(idx)).collect();
                feature = elementwise_multiply(&col, &feature);
            }
            let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                .map_err(|_| FeatureError::MatrixCreationFailed)?
                .transpose();
            x = x.h_stack(&new_column);
        }
    }
    Ok(x)
}
