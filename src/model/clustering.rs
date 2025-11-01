//! Implementation of clustering model training.

use crate::model::{ModelError, ModelResult};
use crate::settings::{ClusteringAlgorithmName, ClusteringSettings};
use crate::{
    algorithms::ClusteringAlgorithm,
    metrics::{ClusterMetrics, HCVScore},
};
use comfy_table::{
    Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
};
use smartcore::linalg::basic::arrays::{Array1, Array2};
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};
use std::fmt::{Display, Formatter};

/// Trains clustering models
pub struct ClusteringModel<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number + Ord,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// Settings for the model.
    settings: ClusteringSettings,
    /// Training data.
    x_train: InputArray,
    /// Trained clustering algorithms.
    trained_algorithms: Vec<TrainedClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>>,
}

impl<INPUT, CLUSTER, InputArray, ClusterArray>
    ClusteringModel<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number + Ord,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    /// Create a new clustering model.
    pub fn new(x: InputArray, settings: ClusteringSettings) -> Self {
        Self {
            settings,
            x_train: x,
            trained_algorithms: Vec::new(),
        }
    }

    /// Train the model using every configured algorithm.
    pub fn train(&mut self) {
        self.trained_algorithms.clear();

        for algorithm_name in self.settings.selected_algorithms() {
            let algorithm = ClusteringAlgorithm::from_name(algorithm_name);
            let fitted = algorithm.fit(&self.x_train, &self.settings);
            self.trained_algorithms
                .push(TrainedClusteringAlgorithm::new(algorithm_name, fitted));
        }
    }

    /// Retrieve the trained algorithm identifiers in order of training.
    #[must_use]
    pub fn trained_algorithm_names(&self) -> Vec<ClusteringAlgorithmName> {
        self.trained_algorithms
            .iter()
            .map(|entry| entry.algorithm_name)
            .collect()
    }

    /// Predict cluster assignments for new data using the first trained algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotTrained`] if the model has not been trained.
    pub fn predict(&self, x: &InputArray) -> ModelResult<ClusterArray> {
        let algorithm = self
            .trained_algorithms
            .first()
            .ok_or(ModelError::NotTrained)?;
        algorithm.predict(x, &self.settings)
    }

    /// Predict cluster assignments with a specific algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotTrained`] if the requested algorithm has not been trained.
    pub fn predict_with(
        &self,
        algorithm: ClusteringAlgorithmName,
        x: &InputArray,
    ) -> ModelResult<ClusterArray> {
        let trained = self
            .trained_algorithms
            .iter()
            .find(|entry| entry.algorithm_name == algorithm)
            .ok_or(ModelError::NotTrained)?;
        trained.predict(x, &self.settings)
    }

    /// Evaluate clustering results against known labels.
    ///
    /// # Arguments
    /// * `truth` - Ground truth cluster labels.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been trained.
    pub fn evaluate(&mut self, truth: &ClusterArray) {
        for trained in &mut self.trained_algorithms {
            let predicted = trained
                .predict(&self.x_train, &self.settings)
                .expect("model must be trained before evaluation");
            let mut scores = ClusterMetrics::<CLUSTER>::hcv_score();
            scores.compute(truth, &predicted);
            trained.metrics = Some(scores);
        }
    }
}

impl<INPUT, CLUSTER, InputArray, ClusterArray> Display
    for ClusteringModel<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number + Ord,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec![
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new("Homogeneity").add_attribute(Attribute::Bold),
            Cell::new("Completeness").add_attribute(Attribute::Bold),
            Cell::new("V-Measure").add_attribute(Attribute::Bold),
        ]);

        if self.trained_algorithms.is_empty() {
            table.add_row(vec![
                "Untrained".to_string(),
                "-".to_string(),
                "-".to_string(),
                "-".to_string(),
            ]);
        } else {
            for entry in &self.trained_algorithms {
                table.add_row(entry.display_row());
            }
        }

        write!(f, "{table}")
    }
}

/// Trained clustering algorithm with optional metrics.
struct TrainedClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number + Ord,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    algorithm_name: ClusteringAlgorithmName,
    algorithm: ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>,
    metrics: Option<HCVScore<CLUSTER>>,
}

impl<INPUT, CLUSTER, InputArray, ClusterArray>
    TrainedClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>
where
    INPUT: RealNumber + FloatNumber,
    CLUSTER: Number + Ord,
    InputArray: Array2<INPUT> + Clone,
    ClusterArray: Array1<CLUSTER> + Clone + std::iter::FromIterator<CLUSTER>,
{
    fn new(
        algorithm_name: ClusteringAlgorithmName,
        algorithm: ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>,
    ) -> Self {
        Self {
            algorithm_name,
            algorithm,
            metrics: None,
        }
    }

    fn predict(&self, x: &InputArray, settings: &ClusteringSettings) -> ModelResult<ClusterArray> {
        self.algorithm.predict(x, settings)
    }

    fn display_row(&self) -> Vec<String> {
        let (homogeneity, completeness, v_measure) = if let Some(scores) = &self.metrics {
            let format_score = |s: Option<f64>| match s {
                Some(val) => format!("{val:.2}"),
                None => "-".to_string(),
            };
            (
                format_score(scores.homogeneity()),
                format_score(scores.completeness()),
                format_score(scores.v_measure()),
            )
        } else {
            ("-".to_string(), "-".to_string(), "-".to_string())
        };

        vec![
            self.algorithm_name.to_string(),
            homogeneity,
            completeness,
            v_measure,
        ]
    }
}

