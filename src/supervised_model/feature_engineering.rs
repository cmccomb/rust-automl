//! Feature engineering routines.

use super::SupervisedModel;
use crate::utils::elementwise_multiply;
use itertools::Itertools;
use smartcore::linalg::basic::arrays::{Array, Array1, Array2, MutArrayView1};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::numbers::{floatnum::FloatNumber, realnum::RealNumber};

impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Get interaction features for the data.
    ///
    /// # Arguments
    pub(super) fn interaction_features(mut x: InputArray) -> InputArray {
        let (height, width) = x.shape();
        for column_1 in 0..width {
            for column_2 in (column_1 + 1)..width {
                let col1: Vec<INPUT> = (0..height)
                    .map(|idx| x.get_col(column_1).get(idx).clone())
                    .collect();
                let col2: Vec<INPUT> = (0..height)
                    .map(|idx| x.get_col(column_2).get(idx).clone())
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

    /// Get polynomial features for the data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `order` - The order of the polynomial
    ///
    /// # Returns
    ///
    /// * The data with polynomial features
    pub(super) fn polynomial_features(mut x: InputArray, order: usize) -> InputArray {
        // Get the shape of the matrix
        let (height, width) = x.shape();

        // For each order, get the combinations of columns with replacement
        for n in 2..=order {
            // Get combinations of columns with replacement
            let combinations = (0..width).combinations_with_replacement(n);

            // For each combination, multiply the columns together and add to the matrix
            for combo in combinations {
                // Start with a vector of ones
                let mut feature: Vec<INPUT> = vec![INPUT::one(); height];

                // Multiply the columns together
                for column in combo {
                    let col: Vec<INPUT> = (0..height)
                        .map(|idx| x.get_col(column).get(idx).clone())
                        .collect();
                    feature = elementwise_multiply(&col, &feature);
                }

                // Add the new column to the matrix
                let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                    .expect("Cannot create matrix")
                    .transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }
}
