//! Utility modules for the crate.

pub mod display;
pub mod distance;
pub mod kernels;
pub mod math;

pub use display::{
    debug_option, print_knn_search_algorithm, print_knn_weight_function, print_option,
};
pub use distance::Distance;
pub use kernels::Kernel;
pub use math::{elementwise_multiply, regression_testing_data};
