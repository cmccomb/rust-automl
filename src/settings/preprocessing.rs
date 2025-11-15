//! Preprocessing configuration utilities.
//!
//! This module defines a small DSL for constructing preprocessing pipelines.
//! Pipelines are expressed as ordered lists of [`PreprocessingStep`] values and
//! can be attached to any [`SupervisedSettings`](crate::settings::SupervisedSettings)
//! via the builder helpers.

use core::iter::FromIterator;
use serde::{Deserialize, Serialize};

/// Column selection helpers for preprocessing steps.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnSelector {
    /// Apply to every column.
    #[default]
    All,
    /// Apply only to the listed column indices.
    Include(Vec<usize>),
    /// Apply to every column except the provided indices.
    Exclude(Vec<usize>),
}

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Flexible numeric scaling strategies.
    Scale(ScaleParams),
    /// Missing-value imputation strategies.
    Impute(ImputeParams),
    /// Encode categorical columns into numerical representations.
    EncodeCategorical(CategoricalEncoderParams),
    /// Apply power transformations such as log or Box-Cox.
    PowerTransform(PowerTransformParams),
    /// Filter columns from the feature matrix.
    FilterColumns(ColumnFilterParams),
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
            Self::Scale(params) => write!(f, "Scaled features using {:?}", params.strategy),
            Self::Impute(params) => write!(f, "Imputed features using {:?}", params.strategy),
            Self::EncodeCategorical(params) => {
                write!(f, "Encoded categorical columns with {:?}", params.encoding)
            }
            Self::PowerTransform(params) => {
                write!(f, "Applied power transform {:?}", params.transform)
            }
            Self::FilterColumns(params) => {
                let mode = if params.retain_selected {
                    "retain"
                } else {
                    "drop"
                };
                write!(f, "Column filter ({mode})")
            }
        }
    }
}

/// Configuration for flexible numeric scaling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleParams {
    /// Scaling strategy to apply.
    pub strategy: ScaleStrategy,
    /// Columns to scale.
    pub selector: ColumnSelector,
}

impl Default for ScaleParams {
    fn default() -> Self {
        Self {
            strategy: ScaleStrategy::Standard(StandardizeParams::default()),
            selector: ColumnSelector::All,
        }
    }
}

/// Supported scaling strategies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScaleStrategy {
    /// Standardize (z-score) scaling.
    Standard(StandardizeParams),
    /// Min-max scaling to a feature range.
    MinMax(MinMaxParams),
    /// Robust scaling using medians and IQR.
    Robust(RobustScaleParams),
}

/// Parameters for min-max scaling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinMaxParams {
    /// Target feature range (min, max).
    pub feature_range: (f64, f64),
}

impl Default for MinMaxParams {
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
        }
    }
}

impl MinMaxParams {
    /// Customize the feature range.
    #[must_use]
    pub const fn with_feature_range(mut self, range: (f64, f64)) -> Self {
        self.feature_range = range;
        self
    }
}

/// Parameters for robust scaling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustScaleParams {
    /// Quantile range (e.g. 25th–75th percentiles).
    pub quantile_range: (f64, f64),
}

impl Default for RobustScaleParams {
    fn default() -> Self {
        Self {
            quantile_range: (25.0, 75.0),
        }
    }
}

impl RobustScaleParams {
    /// Customize the quantile range.
    #[must_use]
    pub const fn with_quantile_range(mut self, range: (f64, f64)) -> Self {
        self.quantile_range = range;
        self
    }
}

/// Parameters for missing-value imputation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImputeParams {
    /// Imputation strategy.
    pub strategy: ImputeStrategy,
    /// Columns to impute.
    pub selector: ColumnSelector,
}

impl Default for ImputeParams {
    fn default() -> Self {
        Self {
            strategy: ImputeStrategy::Mean,
            selector: ColumnSelector::All,
        }
    }
}

/// Strategies for missing-value imputation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputeStrategy {
    /// Replace missing values with the column mean.
    Mean,
    /// Replace missing values with the column median.
    Median,
    /// Replace missing values with the most frequent value.
    MostFrequent,
}

/// Parameters for categorical encoding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoricalEncoderParams {
    /// Columns to encode.
    pub selector: ColumnSelector,
    /// Encoding strategy.
    pub encoding: CategoricalEncoding,
}

impl Default for CategoricalEncoderParams {
    fn default() -> Self {
        Self {
            selector: ColumnSelector::All,
            encoding: CategoricalEncoding::Ordinal,
        }
    }
}

/// Supported categorical encoding strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CategoricalEncoding {
    /// Replace categories with ordinal indices.
    Ordinal,
    /// Replace categories with one-hot columns.
    OneHot {
        /// Whether to drop the first generated column to avoid collinearity.
        drop_first: bool,
    },
}

impl CategoricalEncoding {
    /// Convenience constructor for one-hot encoding.
    #[must_use]
    pub const fn one_hot(drop_first: bool) -> Self {
        Self::OneHot { drop_first }
    }
}

/// Parameters for power transformations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerTransformParams {
    /// Columns to transform.
    pub selector: ColumnSelector,
    /// Transform to apply.
    pub transform: PowerTransform,
}

impl Default for PowerTransformParams {
    fn default() -> Self {
        Self {
            selector: ColumnSelector::All,
            transform: PowerTransform::Log { offset: 0.0 },
        }
    }
}

/// Supported power transformations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PowerTransform {
    /// Natural logarithm (with optional offset to ensure positivity).
    Log {
        /// Offset added prior to the logarithm.
        offset: f64,
    },
    /// Box-Cox transformation with configurable lambda.
    BoxCox {
        /// Lambda parameter for the transform.
        lambda: f64,
    },
}

/// Parameters for filtering columns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnFilterParams {
    /// Which columns to consider.
    pub selector: ColumnSelector,
    /// Whether to retain (`true`) or drop (`false`) the selected columns.
    pub retain_selected: bool,
}

impl Default for ColumnFilterParams {
    fn default() -> Self {
        Self {
            selector: ColumnSelector::All,
            retain_selected: true,
        }
    }
}

/// Ordered collection of preprocessing steps.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
