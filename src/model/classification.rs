//! Type alias for classification models using [`SupervisedModel`].

use crate::algorithms::ClassificationAlgorithm;
use crate::settings::ClassificationSettings;

use super::supervised::SupervisedModel;

/// Convenient alias for a supervised classification model.
pub type ClassificationModel<INPUT, OUTPUT, InputArray, OutputArray> = SupervisedModel<
    ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ClassificationSettings,
    InputArray,
    OutputArray,
>;
