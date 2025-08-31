//! Classification algorithm definitions and helpers.

use std::fmt::{Display, Formatter};
use std::time::Instant;

use crate::model::ComparisonEntry;

use super::ClassificationSettings;
use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::basenum::Number;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

/// Supported classification algorithms.
pub enum ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    /// Decision tree classifier
    DecisionTreeClassifier(
        smartcore::tree::decision_tree_classifier::DecisionTreeClassifier<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// K-nearest neighbours classifier with Euclidean distance
    KNNClassifier(
        smartcore::neighbors::knn_classifier::KNNClassifier<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            Euclidian<INPUT>,
        >,
    ),
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    /// Fit the model
    pub(crate) fn fit(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> Self {
        match self {
            Self::DecisionTreeClassifier(_) => Self::DecisionTreeClassifier(
                smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                    x,
                    y,
                    settings
                        .decision_tree_classifier_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::KNNClassifier(_) => Self::KNNClassifier(
                smartcore::neighbors::knn_classifier::KNNClassifier::fit(
                    x,
                    y,
                    smartcore::neighbors::knn_classifier::KNNClassifierParameters::default()
                        .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Euclidian::new()),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
        }
    }

    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> (
        CrossValidationResult,
        ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) {
        match self {
            Self::DecisionTreeClassifier(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::new(),
                    x,
                    y,
                    settings
                        .decision_tree_classifier_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric::<OUTPUT, OutputArray>(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Self::default_decision_tree_classifier().fit(x, y, settings),
            ),
            Self::KNNClassifier(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::neighbors::knn_classifier::KNNClassifier::new(),
                    x,
                    y,
                    smartcore::neighbors::knn_classifier::KNNClassifierParameters::default()
                        .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Euclidian::new()),
                    &settings.get_kfolds(),
                    &settings.get_metric::<OUTPUT, OutputArray>(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Self::default_knn_classifier().fit(x, y, settings),
            ),
        }
    }

    pub(crate) fn cross_validate_model(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> ComparisonEntry<ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>> {
        let start = Instant::now();
        let results = self.cv(x, y, settings);
        let end = Instant::now();
        ComparisonEntry {
            result: results.0,
            algorithm: results.1,
            duration: end.duration_since(start),
        }
    }

    /// Get a vector of all possible algorithms
    pub fn all_algorithms(settings: &ClassificationSettings) -> Vec<Self> {
        let mut algorithms = vec![Self::default_decision_tree_classifier()];
        if settings.knn_classifier_settings.is_some() {
            algorithms.push(Self::default_knn_classifier());
        }
        algorithms
    }

    /// Default decision tree classifier algorithm
    #[must_use]
    pub fn default_decision_tree_classifier() -> Self {
        Self::DecisionTreeClassifier(
            smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::new(),
        )
    }

    /// Default KNN classifier algorithm
    #[must_use]
    pub fn default_knn_classifier() -> Self {
        Self::KNNClassifier(smartcore::neighbors::knn_classifier::KNNClassifier::new())
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> PartialEq
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn eq(&self, other: &Self) -> bool {
        matches!(self, Self::DecisionTreeClassifier(_))
            && matches!(other, Self::DecisionTreeClassifier(_))
            || matches!(self, Self::KNNClassifier(_)) && matches!(other, Self::KNNClassifier(_))
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn default() -> Self {
        Self::default_decision_tree_classifier()
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecisionTreeClassifier(_) => write!(f, "Decision Tree Classifier"),
            Self::KNNClassifier(_) => write!(f, "KNN Classifier"),
        }
    }
}
