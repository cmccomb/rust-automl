use smartcore::api::SupervisedEstimator;
use smartcore::error::Failed;
use smartcore::linalg::basic::arrays::{Array1, Array2};
use smartcore::model_selection::{CrossValidationResult, KFold};

/// Trait encapsulating shared training logic for supervised algorithms.
pub trait SupervisedTrain<INPUT, OUTPUT, InputArray, OutputArray, Settings>
where
    INPUT: smartcore::numbers::realnum::RealNumber
        + smartcore::numbers::basenum::Number
        + Copy
        + std::fmt::Debug
        + std::fmt::Display,
    OUTPUT: smartcore::numbers::basenum::Number + Copy + std::fmt::Debug + std::fmt::Display,
    InputArray: Clone + Array2<INPUT>,
    OutputArray: Clone + Array1<OUTPUT>,
{
    /// Fit the algorithm using provided settings.
    #[allow(clippy::missing_errors_doc)]
    fn fit_inner(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &Settings,
    ) -> Result<Self, Failed>
    where
        Self: Sized;

    /// Perform cross-validation for the algorithm.
    #[allow(clippy::missing_errors_doc)]
    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &Settings,
    ) -> Result<(CrossValidationResult, Self), Failed>
    where
        Self: Sized;

    /// Retrieve the metric function for evaluation.
    fn metric(settings: &Settings) -> fn(&OutputArray, &OutputArray) -> f64;

    /// Shared implementation of cross-validation and fitting.
    #[allow(clippy::too_many_arguments, clippy::missing_errors_doc)]
    fn cross_validate_with<E, P>(
        self,
        estimator: E,
        params: P,
        x: &InputArray,
        y: &OutputArray,
        settings: &Settings,
        kfold: &KFold,
        metric: fn(&OutputArray, &OutputArray) -> f64,
    ) -> Result<(CrossValidationResult, Self), Failed>
    where
        Self: Sized,
        E: SupervisedEstimator<InputArray, OutputArray, P>,
        P: Clone,
    {
        let result =
            smartcore::model_selection::cross_validate(estimator, x, y, params, kfold, &metric)?;
        let model = self.fit_inner(x, y, settings)?;
        Ok((result, model))
    }

    /// Convenience wrapper around [`Self::fit_inner`].
    #[allow(clippy::missing_errors_doc)]
    fn fit(self, x: &InputArray, y: &OutputArray, settings: &Settings) -> Result<Self, Failed>
    where
        Self: Sized,
    {
        self.fit_inner(x, y, settings)
    }
}
