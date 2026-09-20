//! Generic supervised model implementation.

use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    fs::File,
    io::BufReader,
    path::Path,
};

use crate::model::{
    comparison::ComparisonEntry,
    error::{ModelError, ModelResult},
    persistence::{PersistenceError, PersistenceResult, write_atomic},
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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use smartcore::error::{Failed, FailedError};
use smartcore::linalg::{
    basic::arrays::{Array, Array1, Array2, MutArrayView1},
    traits::{
        cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
        svd::SVDDecomposable,
    },
};
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};

const MODEL_FORMAT: &str = "automl-supervised-model";
const MODEL_FORMAT_VERSION: u32 = 1;
const SMARTCORE_MODEL_VERSION: &str = "0.4.2";

#[derive(Serialize)]
struct ModelArtifactRef<'a, A, S, P> {
    format: &'static str,
    version: u32,
    crate_version: &'static str,
    smartcore_version: &'static str,
    settings: &'a S,
    algorithm: &'a A,
    preprocessor: &'a P,
}

#[derive(Deserialize)]
struct ModelArtifact<A, S, P> {
    settings: S,
    algorithm: A,
    preprocessor: P,
}

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

    /// Explain why this trained algorithm cannot be persisted, if applicable.
    ///
    /// Algorithms are persistable by default when their type implements Serde's
    /// serialization traits. Implementations can override this method when an
    /// upstream model omits required state from its serialized representation.
    fn persistence_error(&self) -> Option<String> {
        None
    }

    /// Validate inference state after it has been restored from an artifact.
    ///
    /// Implementations backed by upstream types with private fitted fields can
    /// inspect `encoded` at the versioned serialization boundary. The default
    /// accepts deserialized state without additional checks.
    ///
    /// # Errors
    /// Returns a description when the stored inference state is invalid.
    fn validate_persisted(_encoded: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }
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
    /// Inference algorithm restored from a persisted artifact.
    restored_algorithm: Option<A>,
    /// Whether the model still owns the raw data required for training.
    has_training_data: bool,
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
            restored_algorithm: None,
            has_training_data: true,
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
    /// `2..=number_of_training_rows`, preprocessing or an algorithm fails, or the
    /// model was loaded without training data.
    pub fn train(&mut self) -> Result<(), Failed> {
        if !self.has_training_data {
            return Err(Failed::fit(
                "a loaded model is inference-only and does not contain training data",
            ));
        }
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
            FinalAlgorithm::Best => self
                .inference_algorithm()?
                .predict(&x)
                .map_err(|e| ModelError::Inference(e.to_string())),
        }
    }

    /// Save the trained inference model and fitted preprocessing state as JSON.
    ///
    /// The artifact omits this model wrapper's separate training buffers and
    /// cross-validation leaderboard. It still stores all inference state owned
    /// by the winning estimator, which can include training-derived or original
    /// examples. KNN models retain training rows and targets, for example, and
    /// SVR models retain support vectors. Treat persisted artifacts as
    /// potentially sensitive. A model returned by [`Self::load`] can run
    /// inference and be saved again, but cannot be retrained. Writes use an
    /// atomic same-directory replacement. On Unix, new artifacts default to
    /// mode `0600` and overwrites retain the destination's existing mode.
    ///
    /// # Errors
    ///
    /// Returns [`PersistenceError::NotTrained`] when no final model is available,
    /// [`PersistenceError::UnsupportedAlgorithm`] when the winning algorithm does
    /// not expose serializable inference state, or a persistence error when the
    /// artifact cannot be encoded or written.
    pub fn save<P>(&self, path: P) -> PersistenceResult<()>
    where
        P: AsRef<Path>,
        A: Serialize,
        S: Serialize,
        Preprocessor<A::Input, InputArray>: Serialize,
    {
        let algorithm = self
            .inference_algorithm()
            .map_err(|_| PersistenceError::NotTrained)?;
        if let Some(reason) = algorithm.persistence_error() {
            return Err(PersistenceError::UnsupportedAlgorithm(reason));
        }

        let artifact = ModelArtifactRef {
            format: MODEL_FORMAT,
            version: MODEL_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION"),
            smartcore_version: SMARTCORE_MODEL_VERSION,
            settings: &self.settings,
            algorithm,
            preprocessor: &self.preprocessor,
        };
        let path = path.as_ref();
        write_atomic(path, &artifact)
    }

    /// Load a versioned JSON model artifact for inference.
    ///
    /// Loaded models omit the wrapper's separate training buffers and
    /// cross-validation results, but fitted estimator state can retain examples
    /// required for inference. Use [`Self::predict`] for inference; construct a
    /// new model to train again. The format and common required fitted fields
    /// are validated, but artifacts are not safe for untrusted or adversarial
    /// input. Only load files from a trusted source.
    ///
    /// # Errors
    ///
    /// Returns a persistence error when the file cannot be read or decoded,
    /// [`PersistenceError::InvalidFormat`] for a different artifact type, or
    /// [`PersistenceError::UnsupportedVersion`] for an incompatible version.
    pub fn load<P>(path: P) -> PersistenceResult<Self>
    where
        P: AsRef<Path>,
        A: DeserializeOwned,
        S: DeserializeOwned,
        Preprocessor<A::Input, InputArray>: DeserializeOwned,
    {
        let path = path.as_ref();
        let file = File::open(path).map_err(|err| PersistenceError::Io {
            operation: "open model artifact",
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        let encoded_value: serde_json::Value = serde_json::from_reader(BufReader::new(file))
            .map_err(|err| {
                if err.is_io() {
                    PersistenceError::Io {
                        operation: "read model artifact",
                        path: path.to_path_buf(),
                        message: err.to_string(),
                    }
                } else {
                    PersistenceError::Decode(err.to_string())
                }
            })?;
        let format = encoded_value
            .get("format")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PersistenceError::InvalidFormat("artifact is missing a format tag".to_string())
            })?;
        if format != MODEL_FORMAT {
            return Err(PersistenceError::InvalidFormat(format!(
                "expected {MODEL_FORMAT:?}, found {format:?}"
            )));
        }
        let version = encoded_value
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| {
                PersistenceError::InvalidFormat("artifact has no valid format version".to_string())
            })?;
        if version != MODEL_FORMAT_VERSION {
            return Err(PersistenceError::UnsupportedVersion {
                supported: MODEL_FORMAT_VERSION,
                found: version,
            });
        }
        let smartcore_version = encoded_value
            .get("smartcore_version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PersistenceError::InvalidFormat(
                    "artifact is missing a SmartCore model version".to_string(),
                )
            })?;
        if smartcore_version != SMARTCORE_MODEL_VERSION {
            return Err(PersistenceError::InvalidFormat(format!(
                "expected SmartCore model version {SMARTCORE_MODEL_VERSION}, found {smartcore_version}"
            )));
        }

        let algorithm_value = encoded_value.get("algorithm").ok_or_else(|| {
            PersistenceError::InvalidFormat("artifact is missing algorithm state".to_string())
        })?;
        let preprocessor_value = encoded_value.get("preprocessor").ok_or_else(|| {
            PersistenceError::InvalidFormat(
                "artifact is missing fitted preprocessing state".to_string(),
            )
        })?;
        A::validate_persisted(algorithm_value).map_err(PersistenceError::InvalidFormat)?;
        Preprocessor::<A::Input, InputArray>::validate_persisted(preprocessor_value)
            .map_err(PersistenceError::InvalidFormat)?;
        let artifact: ModelArtifact<A, S, Preprocessor<A::Input, InputArray>> =
            serde_json::from_value(encoded_value)
                .map_err(|err| PersistenceError::Decode(err.to_string()))?;
        let empty_features = InputArray::zeros(0, 0);
        Ok(Self {
            settings: artifact.settings,
            x_train_raw: empty_features.clone(),
            x_train: empty_features,
            y_train: OutputArray::zeros(0),
            comparison: Vec::new(),
            comparison_metric: None,
            preprocessor: artifact.preprocessor,
            restored_algorithm: Some(artifact.algorithm),
            has_training_data: false,
        })
    }

    fn inference_algorithm(&self) -> ModelResult<&A> {
        if !matches!(
            self.settings.supervised().final_model_approach,
            FinalAlgorithm::Best
        ) {
            return Err(ModelError::NotTrained);
        }
        self.restored_algorithm
            .as_ref()
            .or_else(|| self.comparison.first().map(|entry| &entry.algorithm))
            .ok_or(ModelError::NotTrained)
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
