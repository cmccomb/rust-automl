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
