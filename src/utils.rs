use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::neighbors::KNNWeightFunction;
use std::fmt::{Display, Formatter};

#[derive(PartialEq)]
pub(crate) enum Status {
    Starting,
    DataLoaded,
    ModelsCompared,
    FinalModelTrained,
}

pub(crate) fn print_option<T: Display>(x: Option<T>) -> String {
    match x {
        None => "None".to_string(),
        Some(y) => format!("{}", y),
    }
}

pub(crate) fn print_knn_weight_function(f: &KNNWeightFunction) -> String {
    match f {
        KNNWeightFunction::Uniform => "Uniform".to_string(),
        KNNWeightFunction::Distance => "Distance".to_string(),
    }
}

pub(crate) fn print_knn_search_algorithm(a: &KNNAlgorithmName) -> String {
    match a {
        KNNAlgorithmName::LinearSearch => "Linear Search".to_string(),
        KNNAlgorithmName::CoverTree => "Cover Tree".to_string(),
    }
}

/// Kernel options for use with support vector machines
pub enum Kernel {
    /// Linear Kernel
    Linear,

    /// Polynomial kernel
    Polynomial(f32, f32, f32),

    /// Radial basis function kernel
    RBF(f32),

    /// Sigmoid kernel
    Sigmoid(f32, f32),
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernel::Linear => write!(f, "Linear"),
            Kernel::Polynomial(degree, gamma, coef) => write!(
                f,
                "Polynomial\n    degree = {}\n    gamma = {}\n    coef = {}",
                degree, gamma, coef
            ),
            Kernel::RBF(gamma) => write!(f, "RBF\n    gamma = {}", gamma),
            Kernel::Sigmoid(gamma, coef) => {
                write!(f, "Sigmoid\n    gamma = {}\n    coef = {}", gamma, coef)
            }
        }
    }
}

/// Distance metrics
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
            Distance::Euclidean => write!(f, "Euclidean"),
            Distance::Manhattan => write!(f, "Manhattan"),
            Distance::Minkowski(n) => write!(f, "Minkowski\n    p = {}", n),
            Distance::Mahalanobis => write!(f, "Mahalanobis"),
            Distance::Hamming => write!(f, "Hamming"),
        }
    }
}
