use automl::utils::{
    display::{debug_option, print_knn_search_algorithm, print_knn_weight_function, print_option},
    math::{elementwise_multiply, regression_testing_data},
};
use smartcore::{
    algorithm::neighbour::KNNAlgorithmName, linalg::basic::arrays::Array,
    neighbors::KNNWeightFunction,
};

#[test]
fn test_print_option_and_debug_option() {
    // Arrange
    let some = Some(5);
    let none: Option<i32> = None;

    // Act
    let printed_some = print_option(some);
    let printed_none = print_option(none);
    let debugged_none = debug_option::<i32>(None);

    // Assert
    assert_eq!(printed_some, "5");
    assert_eq!(printed_none, "None");
    assert_eq!(debugged_none, "None");
}

#[test]
fn test_knn_display_helpers() {
    // Arrange
    let weight_fn = KNNWeightFunction::Uniform;
    let search_alg = KNNAlgorithmName::LinearSearch;

    // Act
    let weight_name = print_knn_weight_function(&weight_fn);
    let alg_name = print_knn_search_algorithm(&search_alg);

    // Assert
    assert_eq!(weight_name, "Uniform");
    assert_eq!(alg_name, "Linear Search");
}

#[test]
fn test_math_helpers() {
    // Arrange
    let v1 = [1, 2, 3];
    let v2 = [4, 5, 6];

    // Act
    let result = elementwise_multiply(&v1, &v2);
    let (x, y) = regression_testing_data();

    // Assert
    assert_eq!(result, vec![4, 10, 18]);
    assert_eq!(x.shape(), (16, 6));
    assert_eq!(y.len(), 16);
}
