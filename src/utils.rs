use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};
use std::fmt::{Debug, Display, Formatter};

pub(crate) fn print_option<T: Display>(x: Option<T>) -> String {
    match x {
        None => "None".to_string(),
        Some(y) => format!("{}", y),
    }
}
pub(crate) fn debug_option<T: Debug>(x: Option<T>) -> String {
    match x {
        None => "None".to_string(),
        Some(y) => format!("{:#?}", y),
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
#[derive(serde::Serialize, serde::Deserialize)]
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
            Distance::Euclidean => write!(f, "Euclidean"),
            Distance::Manhattan => write!(f, "Manhattan"),
            Distance::Minkowski(n) => write!(f, "Minkowski\n    p = {}", n),
            Distance::Mahalanobis => write!(f, "Mahalanobis"),
            Distance::Hamming => write!(f, "Hamming"),
        }
    }
}

/// Function to do element-wise multiplication fo two vectors
pub fn elementwise_multiply(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    v1.iter().zip(v2).map(|(&i1, &i2)| i1 * i2).collect()
}

#[cfg(any(feature = "csv"))]
use polars::prelude::{CsvReader, DataFrame, SerReader};

#[cfg(any(feature = "csv"))]
pub(crate) fn validate_and_read(file_path: &str, header: bool) -> DataFrame {
    match CsvReader::from_path(file_path) {
        Ok(csv) => csv
            .infer_schema(None)
            .has_header(header)
            .finish()
            .expect("Cannot read file as CSV"),
        Err(_) => {
            if let Ok(_) = url::Url::parse(file_path) {
                let file_contents = minreq::get(file_path).send().unwrap();
                let temp = temp_file::with_contents(file_contents.as_bytes());

                CsvReader::from_path(temp.path())
                    .expect("Cannot find file")
                    .infer_schema(Some(10))
                    .has_header(header)
                    .finish()
                    .expect("Cannot read file as CSV")
            } else {
                panic!("The string {} is not a valid URL or file path.", file_path)
            }
        }
    }
}
