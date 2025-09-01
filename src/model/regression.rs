//! Type alias for regression models using [`SupervisedModel`].

use crate::algorithms::RegressionAlgorithm;
use crate::settings::RegressionSettings;

use super::supervised::SupervisedModel;

/// Convenient alias for a supervised regression model.
pub type RegressionModel<INPUT, OUTPUT, InputArray, OutputArray> = SupervisedModel<
    RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    InputArray,
    OutputArray,
>;
