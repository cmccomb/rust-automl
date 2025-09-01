use automl::utils::display::{
    debug_option, print_knn_search_algorithm, print_knn_weight_function, print_option,
};
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::neighbors::KNNWeightFunction;

#[test]
fn print_option_returns_value_for_some() {
    // Arrange
    let value = Some(10);

    // Act
    let output = print_option(value);

    // Assert
    assert_eq!(output, "10");
}

#[test]
fn print_option_returns_none_for_none() {
    // Arrange
    let value: Option<i32> = None;

    // Act
    let output = print_option(value);

    // Assert
    assert_eq!(output, "None");
}

#[test]
fn debug_option_formats_value() {
    // Arrange
    let value = Some(42);

    // Act
    let output = debug_option(value);

    // Assert
    assert_eq!(output, "42");
}

#[test]
fn debug_option_formats_complex_value() {
    // Arrange
    let value = Some(vec![1, 2]);

    // Act
    let output = debug_option(value);

    // Assert
    assert_eq!(output, "[\n    1,\n    2,\n]");
}

#[test]
fn debug_option_formats_none() {
    // Arrange
    let value: Option<i32> = None;

    // Act
    let output = debug_option(value);

    // Assert
    assert_eq!(output, "None");
}

#[test]
fn print_knn_weight_function_uniform() {
    // Arrange
    let weight = KNNWeightFunction::Uniform;

    // Act
    let output = print_knn_weight_function(&weight);

    // Assert
    assert_eq!(output, "Uniform");
}

#[test]
fn print_knn_weight_function_distance() {
    // Arrange
    let weight = KNNWeightFunction::Distance;

    // Act
    let output = print_knn_weight_function(&weight);

    // Assert
    assert_eq!(output, "Distance");
}

#[test]
fn print_knn_search_algorithm_linear_search() {
    // Arrange
    let algorithm = KNNAlgorithmName::LinearSearch;

    // Act
    let output = print_knn_search_algorithm(&algorithm);

    // Assert
    assert_eq!(output, "Linear Search");
}

#[test]
fn print_knn_search_algorithm_cover_tree() {
    // Arrange
    let algorithm = KNNAlgorithmName::CoverTree;

    // Act
    let output = print_knn_search_algorithm(&algorithm);

    // Assert
    assert_eq!(output, "Cover Tree");
}
