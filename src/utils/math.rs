//! Mathematical helper functions.

use std::ops::Mul;

/// Function to do element-wise multiplication of two vectors
pub fn elementwise_multiply<T>(v1: &[T], v2: &[T]) -> Vec<T>
where
    T: Mul<Output = T> + Copy,
{
    v1.iter().zip(v2).map(|(&i1, &i2)| i1 * i2).collect()
}
