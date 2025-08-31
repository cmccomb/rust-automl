//! Utility functions for the crate.

pub mod display;
pub mod distance;
pub mod kernels;
pub mod math;

pub use self::display::{
    debug_option, print_knn_search_algorithm, print_knn_weight_function, print_option,
};
pub use self::distance::Distance;
pub use self::kernels::Kernel;
pub use self::math::elementwise_multiply;
