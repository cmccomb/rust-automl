use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::neighbors::KNNWeightFunction;
use std::fmt::Display;

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
