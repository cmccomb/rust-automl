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
    /// The fitted algorithm.
    algorithm: Option<ClusteringAlgorithm<INPUT, CLUSTER, InputArray, ClusterArray>>,
    /// Optional clustering evaluation metrics.
    metrics: Option<HCVScore<CLUSTER>>,
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
            algorithm: None,
            metrics: None,
        }
    }

    /// Train the model using the configured algorithm.
    pub fn train(&mut self) {
        let alg = match self.settings.algorithm {
            ClusteringAlgorithmName::KMeans => ClusteringAlgorithm::default_kmeans(),
            ClusteringAlgorithmName::Agglomerative => ClusteringAlgorithm::default_agglomerative(),
            ClusteringAlgorithmName::DBSCAN => ClusteringAlgorithm::default_dbscan(),
        };
        let fitted = alg.fit(&self.x_train, &self.settings);
        self.algorithm = Some(fitted);
    }

    /// Predict cluster assignments for new data.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotTrained`] if the model has not been trained.
    pub fn predict(&self, x: &InputArray) -> ModelResult<ClusterArray> {
        match &self.algorithm {
            Some(alg) => alg.predict(x, &self.settings),
            None => Err(ModelError::NotTrained),
        }
    }

    /// Evaluate clustering results against known labels.
    ///
    /// # Arguments
    /// * `truth` - Ground truth cluster labels.
    pub fn evaluate(&mut self, truth: &ClusterArray) {
        let predicted = self.predict(&self.x_train);
        let mut scores = ClusterMetrics::<CLUSTER>::hcv_score();
        scores.compute(truth, &predicted);
        self.metrics = Some(scores);
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

        let algorithm = match &self.algorithm {
            Some(alg) => alg.to_string(),
            None => "Untrained".to_string(),
        };

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

        table.add_row(vec![algorithm, homogeneity, completeness, v_measure]);

        write!(f, "{table}")
    }
}
