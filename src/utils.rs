//! Utility functions for the crate.

use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};
use std::fmt::{Debug, Display, Formatter};

/// Convert an Option<T> to a String for printing in display mode.
pub fn print_option<T: Display>(x: Option<T>) -> String {
    x.map_or_else(|| "None".to_string(), |y| format!("{y}"))
}

/// Convert an Option<T> to a String for printing in debug mode.
pub fn debug_option<T: Debug>(x: Option<T>) -> String {
    x.map_or_else(|| "None".to_string(), |y| format!("{y:#?}"))
}

/// Get the name for a knn weight function.
pub fn print_knn_weight_function(f: &KNNWeightFunction) -> String {
    match f {
        KNNWeightFunction::Uniform => "Uniform".to_string(),
        KNNWeightFunction::Distance => "Distance".to_string(),
    }
}

/// Get the name for a knn search algorithm.
pub fn print_knn_search_algorithm(a: &KNNAlgorithmName) -> String {
    match a {
        KNNAlgorithmName::LinearSearch => "Linear Search".to_string(),
        KNNAlgorithmName::CoverTree => "Cover Tree".to_string(),
    }
}

/// Kernel options for use with support vector machines
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Kernel {
    /// Linear Kernel
    Linear,

    /// Polynomial kernel
    Polynomial(f64, f64, f64),

    /// Radial basis function kernel
    RBF(f64),

    /// Sigmoid kernel
    Sigmoid(f64, f64),
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Polynomial(degree, gamma, coef) => write!(
                f,
                "Polynomial\n    degree = {degree}\n    gamma = {gamma}\n    coef = {coef}"
            ),
            Self::RBF(gamma) => write!(f, "RBF\n    gamma = {gamma}"),
            Self::Sigmoid(gamma, coef) => {
                write!(f, "Sigmoid\n    gamma = {gamma}\n    coef = {coef}")
            }
        }
    }
}

/// Distance metrics
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Distance {
    /// Euclidean distance
    Euclidean,

    /// Manhattan distance
    Manhattan,

    /// Minkowski distance, parameterized by p
    Minkowski(u16),

    /// Mahalanobis distance
    Mahalanobis,

    /// Hamming distance
    Hamming,
}

impl Display for Distance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::Manhattan => write!(f, "Manhattan"),
            Self::Minkowski(n) => write!(f, "Minkowski\n    p = {n}"),
            Self::Mahalanobis => write!(f, "Mahalanobis"),
            Self::Hamming => write!(f, "Hamming"),
        }
    }
}

/// Function to do element-wise multiplication fo two vectors
pub fn elementwise_multiply<T>(v1: &[T], v2: &[T]) -> Vec<T>
where
    T: std::ops::Mul<Output = T> + Copy,
{
    v1.iter().zip(v2).map(|(&i1, &i2)| i1 * i2).collect()
}

/// Get some regression data for testing purposes
pub fn regression_testing_data() -> (smartcore::linalg::basic::matrix::DenseMatrix<f64>, Vec<f64>) {
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_array(&[
        &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
        &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
        &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
        &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
        &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
        &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
        &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
        &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
        &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
        &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
        &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
        &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
        &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
        &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
        &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
        &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    ])
    .unwrap();

    let y: Vec<f64> = vec![
        83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2,
        115.7, 116.9,
    ];

    (x, y)
}
