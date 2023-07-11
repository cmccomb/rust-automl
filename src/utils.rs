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
    Polynomial(f32, f32, f32),

    /// Radial basis function kernel
    RBF(f32),

    /// Sigmoid kernel
    Sigmoid(f32, f32),
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
pub fn elementwise_multiply(v1: &[f32], v2: &[f32]) -> Vec<f32> {
    v1.iter().zip(v2).map(|(&i1, &i2)| i1 * i2).collect()
}

#[cfg(any(feature = "csv"))]
use polars::prelude::{CsvReader, DataFrame, PolarsError, SerReader};

#[cfg(any(feature = "csv"))]
/// Read and validate a csv file or URL into a polars `DataFrame`.
pub fn validate_and_read<P>(file_path: P) -> DataFrame
where
    P: AsRef<std::path::Path>,
{
    let file_path_as_str = file_path.as_ref().to_str().unwrap();

    CsvReader::from_path(file_path_as_str).map_or_else(
        |_| {
            if url::Url::parse(file_path_as_str).is_ok() {
                let file_contents = minreq::get(file_path_as_str)
                    .send()
                    .expect("Could not open URL");
                let temp = temp_file::with_contents(file_contents.as_bytes());
                validate_and_read(temp.path().to_str().unwrap())
            } else {
                panic!("The string {file_path_as_str} is not a valid URL or file path.")
            }
        },
        |csv| {
            csv.infer_schema(Some(10))
                .has_header(
                    csv_sniffer::Sniffer::new()
                        .sniff_path(file_path_as_str)
                        .expect("Cannot sniff file")
                        .dialect
                        .header
                        .has_header_row,
                )
                .finish()
                .expect("Cannot read file as CSV")
                .drop_nulls(None)
                .expect("Cannot remove null values")
                .convert_to_float()
                .expect("Cannot convert types")
        },
    )
}

/// Trait to convert to a polars `DataFrame`.
#[cfg(any(feature = "csv"))]
trait Cleanup {
    /// Convert to a polars `DataFrame` with all columns of type float.
    fn convert_to_float(self) -> Result<DataFrame, PolarsError>;
}

#[cfg(any(feature = "csv"))]
impl Cleanup for DataFrame {
    #[allow(unused_mut)]
    fn convert_to_float(mut self) -> Result<DataFrame, PolarsError> {
        // Work in progress
        // for field in self.schema().fields() {
        //     let name = field.name();
        //     if field.data_type().to_string() == "str" {
        //         let ca = self.column(name).unwrap().utf8().unwrap();
        //         let vec_str: Vec<&str> = ca.into_no_null_iter().collect();
        //         let mut unique = vec_str.clone();
        //         unique.sort();
        //         unique.dedup();
        //         let mut new_encoding = vec![0; 0];
        //         if unique.len() == vec_str.len() || unique.len() == 1 {
        //             self.drop_in_place(name);
        //         } else {
        //             vec_str.into_iter().for_each(|x| {
        //                 new_encoding.push(unique.iter().position(|&y| y == x).unwrap() as u64)
        //             });
        //             self.with_column(Series::new(name, &new_encoding));
        //         }
        //     }
        // }

        Ok(self)
    }
}
