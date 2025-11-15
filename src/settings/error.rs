//! Error types used by settings modules.

use std::fmt::{Display, Formatter};

use super::Metric;

/// Errors related to model settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    /// A required metric was not specified.
    MetricNotSet,
    /// The provided metric is not supported for the task.
    UnsupportedMetric(Metric),
    /// Preprocessing configuration failed to run successfully.
    PreProcessingFailed(String),
}

impl Display for SettingsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MetricNotSet => write!(f, "a metric must be set"),
            Self::UnsupportedMetric(m) => write!(f, "unsupported metric: {m}"),
            Self::PreProcessingFailed(msg) => {
                write!(f, "preprocessing configuration failed: {msg}")
            }
        }
    }
}

impl std::error::Error for SettingsError {}
