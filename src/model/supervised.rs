//! Generic supervised model implementation.

use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
};

use crate::model::{
    comparison::ComparisonEntry,
    error::{ModelError, ModelResult},
    preprocessing::Preprocessor,
};
use crate::settings::{
    ClassificationSettings, FinalAlgorithm, Metric, RegressionSettings, SettingsError,
    SupervisedSettings,
};
use comfy_table::{
    Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
};
use humantime::format_duration;
use smartcore::error::{Failed, FailedError};
use smartcore::linalg::{
    basic::arrays::{Array, Array1, Array2, MutArrayView1},
    traits::{
        cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
        svd::SVDDecomposable,
    },
};
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};

/// Trait representing a supervised learning algorithm.
pub trait Algorithm<ASettings>: Sized {
    /// Numeric type for features.
    type Input: RealNumber + FloatNumber;
    /// Numeric type for targets.
    type Output: Number;
    /// Feature matrix type.
    type InputArray: Clone
        + Array<Self::Input, (usize, usize)>
        + Array2<Self::Input>
        + EVDDecomposable<Self::Input>
        + SVDDecomposable<Self::Input>
        + CholeskyDecomposable<Self::Input>
        + QRDecomposable<Self::Input>;
    /// Target vector type.
    type OutputArray: Clone + MutArrayView1<Self::Output> + Array1<Self::Output>;

    /// Predict values for new data.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] if the underlying algorithm cannot produce predictions.
    fn predict(&self, x: &Self::InputArray) -> Result<Self::OutputArray, Failed>;

    /// Perform cross-validation and return a trained model entry.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] if model training or evaluation fails.
    fn cross_validate_model(
        self,
        x: &Self::InputArray,
        y: &Self::OutputArray,
        settings: &ASettings,
    ) -> Result<ComparisonEntry<Self>, Failed>;

    /// Retrieve all algorithm variants available for comparison.
    fn all_algorithms(settings: &ASettings) -> Vec<Self>;
}

/// Accessor for common supervised settings.
pub trait SupervisedLearningSettings {
    /// Get the inner [`SupervisedSettings`].
    fn supervised(&self) -> &SupervisedSettings;
}

impl SupervisedLearningSettings for ClassificationSettings {
    fn supervised(&self) -> &SupervisedSettings {
        &self.supervised
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedLearningSettings
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    fn supervised(&self) -> &SupervisedSettings {
        &self.supervised
    }
}

/// Generic model for supervised algorithms.
pub struct SupervisedModel<A, S, InputArray, OutputArray>
where
    A: Algorithm<S, InputArray = InputArray, OutputArray = OutputArray>,
    S: SupervisedLearningSettings,
    InputArray: Clone
        + Array<A::Input, (usize, usize)>
        + Array2<A::Input>
        + EVDDecomposable<A::Input>
        + SVDDecomposable<A::Input>
        + CholeskyDecomposable<A::Input>
        + QRDecomposable<A::Input>,
    OutputArray: Clone + MutArrayView1<A::Output> + Array1<A::Output>,
{
    /// Settings for the model.
    pub settings: S,
    /// Original training features used to recompute preprocessing steps.
    x_train_raw: InputArray,
    /// Preprocessed training features fed to algorithms.
    x_train: InputArray,
    /// Training targets.
    y_train: OutputArray,
    /// Comparison results for trained models.
    comparison: Vec<ComparisonEntry<A>>,
    /// Metric used by the last successful comparison.
    comparison_metric: Option<Metric>,
    /// Preprocessor for feature engineering.
    preprocessor: Preprocessor<A::Input, InputArray>,
}

impl<A, S, InputArray, OutputArray> SupervisedModel<A, S, InputArray, OutputArray>
where
    A: Algorithm<S, InputArray = InputArray, OutputArray = OutputArray>,
    S: SupervisedLearningSettings,
    InputArray: Clone
        + Array<A::Input, (usize, usize)>
        + Array2<A::Input>
        + EVDDecomposable<A::Input>
        + SVDDecomposable<A::Input>
        + CholeskyDecomposable<A::Input>
        + QRDecomposable<A::Input>,
    OutputArray: Clone + MutArrayView1<A::Output> + Array1<A::Output>,
{
    /// Create a new supervised model.
    pub fn new(x: InputArray, y: OutputArray, settings: S) -> Self {
        let x_train_raw = x.clone();
        Self {
            settings,
            x_train_raw,
            x_train: x,
            y_train: y,
            comparison: Vec::new(),
            comparison_metric: None,
            preprocessor: Preprocessor::new(),
        }
    }

    /// Train all available algorithms and record their performance.
    ///
    /// A successful call replaces the previous comparison and fitted preprocessing
    /// state. If training fails, the last successful model and its scores remain
    /// available; an initially untrained model remains untrained. Changes to
    /// [`Self::settings`] are not rolled back.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] if no algorithms are selected, the fold count is outside
    /// `2..=number_of_training_rows`, preprocessing fails, or any algorithm fails.
    pub fn train(&mut self) -> Result<(), Failed> {
        let sup = self.settings.supervised();
        let rows = self.x_train_raw.shape().0;
        if sup.number_of_folds < 2 || sup.number_of_folds > rows {
            return Err(Failed::because(
                FailedError::ParametersError,
                &format!(
                    "number of folds must be between 2 and the number of training rows ({rows})"
                ),
            ));
        }
        let algorithms = <A>::all_algorithms(&self.settings);
        if algorithms.is_empty() {
            return Err(Failed::because(
                FailedError::ParametersError,
                "no algorithms are selected for training",
            ));
        }

        // Build the complete next run before replacing any fitted state.
        let mut preprocessor = Preprocessor::new();
        let x_train = preprocessor
            .fit_transform(self.x_train_raw.clone(), &sup.preprocessing)
            .map_err(|err| Self::preprocessing_failed(&err))?;
        let comparison = algorithms
            .into_iter()
            .map(|alg| alg.cross_validate_model(&x_train, &self.y_train, &self.settings))
            .collect::<Result<Vec<_>, _>>()?;

        self.comparison_metric = Some(sup.sort_by);
        self.preprocessor = preprocessor;
        self.x_train = x_train;
        self.comparison = comparison;
        self.sort();
        Ok(())
    }

    /// Predict using the best-performing model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotTrained`] if no training run has succeeded or no
    /// final model is requested, or [`ModelError::Inference`] if inference fails.
    pub fn predict(&self, x: InputArray) -> ModelResult<OutputArray> {
        let x = self.preprocessor.preprocess(x)?;

        match self.settings.supervised().final_model_approach {
            FinalAlgorithm::None => Err(ModelError::NotTrained),
            FinalAlgorithm::Best => {
                let entry = self.comparison.first().ok_or(ModelError::NotTrained)?;
                entry
                    .algorithm
                    .predict(&x)
                    .map_err(|e| ModelError::Inference(e.to_string()))
            }
        }
    }

    fn sort(&mut self) {
        let sort_by = &self.settings.supervised().sort_by;
        self.comparison.sort_by(|a, b| {
            a.result
                .mean_test_score()
                .partial_cmp(&b.result.mean_test_score())
                .unwrap_or(Equal)
        });
        if matches!(sort_by, Metric::RSquared | Metric::Accuracy) {
            self.comparison.reverse();
        }
    }

    fn preprocessing_failed(err: &SettingsError) -> Failed {
        Failed::because(FailedError::ParametersError, &err.to_string())
    }
}

impl<A, S, InputArray, OutputArray> Display for SupervisedModel<A, S, InputArray, OutputArray>
where
    A: Algorithm<S, InputArray = InputArray, OutputArray = OutputArray> + Display,
    S: SupervisedLearningSettings,
    InputArray: Clone
        + Array<A::Input, (usize, usize)>
        + Array2<A::Input>
        + EVDDecomposable<A::Input>
        + SVDDecomposable<A::Input>
        + CholeskyDecomposable<A::Input>
        + QRDecomposable<A::Input>,
    OutputArray: Clone + MutArrayView1<A::Output> + Array1<A::Output>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let metric = self
            .comparison_metric
            .unwrap_or(self.settings.supervised().sort_by);
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec![
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new("Time").add_attribute(Attribute::Bold),
            Cell::new(format!("Training {metric}")).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {metric}")).add_attribute(Attribute::Bold),
        ]);

        for entry in &self.comparison {
            let mut row = Vec::new();
            row.push(entry.algorithm.to_string());
            row.push(format_duration(entry.duration).to_string());
            let decider = f64::midpoint(
                entry.result.mean_train_score(),
                entry.result.mean_test_score(),
            )
            .abs();
            if (0.01..1000.0).contains(&decider) {
                row.push(format!("{:.2}", entry.result.mean_train_score()));
                row.push(format!("{:.2}", entry.result.mean_test_score()));
            } else {
                row.push(format!("{:.3e}", entry.result.mean_train_score()));
                row.push(format!("{:.3e}", entry.result.mean_test_score()));
            }
            table.add_row(row);
        }

        write!(f, "{table}")
    }
}
