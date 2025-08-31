//! Utilities for formatting output.

use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};
use std::fmt::{Debug, Display};

/// Convert an `Option<T>` to a `String` for printing in display mode.
///
/// # Examples
/// ```
/// use automl::utils::display::print_option;
/// assert_eq!(print_option(Some(5)), "5");
/// assert_eq!(print_option::<i32>(None), "None");
/// ```
pub fn print_option<T: Display>(x: Option<T>) -> String {
    x.map_or_else(|| "None".to_string(), |y| format!("{y}"))
}

/// Convert an `Option<T>` to a `String` for printing in debug mode.
///
/// # Examples
/// ```
/// use automl::utils::display::debug_option;
/// assert_eq!(debug_option(Some(5)), "5");
/// assert_eq!(debug_option::<i32>(None), "None");
/// ```
pub fn debug_option<T: Debug>(x: Option<T>) -> String {
    x.map_or_else(|| "None".to_string(), |y| format!("{y:#?}"))
}

/// Get the name for a KNN weight function.
///
/// # Examples
/// ```
/// use automl::utils::display::print_knn_weight_function;
/// use smartcore::neighbors::KNNWeightFunction;
/// assert_eq!(print_knn_weight_function(&KNNWeightFunction::Uniform), "Uniform");
/// ```
#[must_use]
pub fn print_knn_weight_function(f: &KNNWeightFunction) -> String {
    match f {
        KNNWeightFunction::Uniform => "Uniform".to_string(),
        KNNWeightFunction::Distance => "Distance".to_string(),
    }
}

/// Get the name for a KNN search algorithm.
///
/// # Examples
/// ```
/// use automl::utils::display::print_knn_search_algorithm;
/// use smartcore::algorithm::neighbour::KNNAlgorithmName;
/// assert_eq!(
///     print_knn_search_algorithm(&KNNAlgorithmName::LinearSearch),
///     "Linear Search"
/// );
/// ```
#[must_use]
pub fn print_knn_search_algorithm(a: &KNNAlgorithmName) -> String {
    match a {
        KNNAlgorithmName::LinearSearch => "Linear Search".to_string(),
        KNNAlgorithmName::CoverTree => "Cover Tree".to_string(),
    }
}
