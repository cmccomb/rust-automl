//! Display utilities for formatting optional and KNN values.

use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};
use std::fmt::{Debug, Display};

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
