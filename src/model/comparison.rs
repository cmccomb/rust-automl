//! Shared comparison entry structure for model evaluations.

use smartcore::model_selection::CrossValidationResult;
use std::time::Duration;

/// Stores the outcome of training an algorithm during model comparison.
#[derive(Debug, Clone)]
pub struct ComparisonEntry<A> {
    /// Cross-validation metrics for the trained algorithm.
    pub result: CrossValidationResult,
    /// The trained algorithm instance.
    pub algorithm: A,
    /// Duration taken to train and evaluate the algorithm.
    pub duration: Duration,
}
