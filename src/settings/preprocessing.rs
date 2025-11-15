//! Preprocessing configuration utilities.
//!
//! This module defines a small DSL for constructing preprocessing pipelines.
//! Pipelines are expressed as ordered lists of [`PreprocessingStep`] values and
//! can be attached to any [`SupervisedSettings`](crate::settings::SupervisedSettings)
//! via the builder helpers.

use core::iter::FromIterator;
use serde::{Deserialize, Serialize};

/// Parameters for standardizing features column-wise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StandardizeParams {
    /// Whether to subtract the mean from each feature.
    pub with_mean: bool,
    /// Whether to divide by the sample standard deviation.
    pub with_std: bool,
}

impl Default for StandardizeParams {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
        }
    }
}

impl StandardizeParams {
    /// Enable or disable centering.
    #[must_use]
    pub const fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Enable or disable scaling by the sample standard deviation.
    #[must_use]
    pub const fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }
}

/// A single preprocessing operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreprocessingStep {
    /// Add pairwise interaction terms.
    AddInteractions,
    /// Add polynomial features up to `order`.
    AddPolynomial {
        /// Maximum order of the generated polynomial features.
        order: usize,
    },
    /// Replace the feature space with the top PCA components.
    ReplaceWithPCA {
        /// Number of PCA components to retain.
        number_of_components: usize,
    },
    /// Replace the feature space with the top SVD components.
    ReplaceWithSVD {
        /// Number of SVD components to retain.
        number_of_components: usize,
    },
    /// Standardize features column-wise.
    Standardize(StandardizeParams),
}

impl core::fmt::Display for PreprocessingStep {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AddInteractions => write!(f, "Interaction terms added"),
            Self::AddPolynomial { order } => {
                write!(f, "Polynomial terms added (order = {order})")
            }
            Self::ReplaceWithPCA {
                number_of_components,
            } => write!(f, "Replaced with PCA features (n = {number_of_components})"),
            Self::ReplaceWithSVD {
                number_of_components,
            } => write!(f, "Replaced with SVD features (n = {number_of_components})"),
            Self::Standardize(params) => write!(
                f,
                "Standardized features (with_mean = {}, with_std = {})",
                params.with_mean, params.with_std
            ),
        }
    }
}

/// Ordered collection of preprocessing steps.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreprocessingPipeline {
    steps: Vec<PreprocessingStep>,
}

impl PreprocessingPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Return true if the pipeline contains no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Immutable view of the configured steps.
    #[must_use]
    pub fn steps(&self) -> &[PreprocessingStep] {
        &self.steps
    }

    /// Add a new step to the end of the pipeline, returning the updated
    /// pipeline for chaining.
    #[must_use]
    pub fn add_step(mut self, step: PreprocessingStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Mutably push a new step to the end of the pipeline.
    pub fn push_step(&mut self, step: PreprocessingStep) {
        self.steps.push(step);
    }
}

impl From<Vec<PreprocessingStep>> for PreprocessingPipeline {
    fn from(steps: Vec<PreprocessingStep>) -> Self {
        Self { steps }
    }
}

impl From<PreprocessingStep> for PreprocessingPipeline {
    fn from(step: PreprocessingStep) -> Self {
        Self { steps: vec![step] }
    }
}

impl FromIterator<PreprocessingStep> for PreprocessingPipeline {
    fn from_iter<T: IntoIterator<Item = PreprocessingStep>>(iter: T) -> Self {
        Self {
            steps: iter.into_iter().collect(),
        }
    }
}

impl core::fmt::Display for PreprocessingPipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.steps.is_empty() {
            return write!(f, "No preprocessing");
        }

        for (idx, step) in self.steps.iter().enumerate() {
            if idx > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{step}")?;
        }
        Ok(())
    }
}
