//! Re-exported clustering metrics.
//!
//! This module exposes clustering evaluation helpers from `smartcore`.
//!
//! # Examples
//! ```
//! use automl::metrics::ClusterMetrics;
//! let y_true = vec![1_u8, 1, 2, 2];
//! let y_pred = vec![1_u8, 1, 2, 2];
//! let mut score = ClusterMetrics::<u8>::hcv_score();
//! score.compute(&y_true, &y_pred);
//! assert_eq!(score.v_measure().unwrap(), 1.0);
//! ```
pub use smartcore::metrics::cluster_hcv::HCVScore;
pub use smartcore::metrics::{
    ClusterMetrics, completeness_score, homogeneity_score, v_measure_score,
};
