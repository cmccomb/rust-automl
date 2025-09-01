//! Classification algorithm definitions and helpers.

use std::fmt::{Display, Formatter};
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::{ClassificationSettings, WithSupervisedSettings};
use smartcore::api::SupervisedEstimator;
use smartcore::error::Failed;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;
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
        + EVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
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
    /// Random forest classifier
    RandomForestClassifier(
        smartcore::ensemble::random_forest_classifier::RandomForestClassifier<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Logistic regression classifier
    LogisticRegression(
        smartcore::linear::logistic_regression::LogisticRegression<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    SupervisedTrain<INPUT, OUTPUT, InputArray, OutputArray, ClassificationSettings>
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
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn fit_inner(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> Result<Self, Failed> {
        Ok(match self {
            Self::DecisionTreeClassifier(_) => Self::DecisionTreeClassifier(
                smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                    x,
                    y,
                    settings
                        .decision_tree_classifier_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                )?,
            ),
            Self::KNNClassifier(_) => {
                Self::KNNClassifier(smartcore::neighbors::knn_classifier::KNNClassifier::fit(
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
                )?)
            }
            Self::RandomForestClassifier(_) => Self::RandomForestClassifier(
                smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                    x,
                    y,
                    settings
                        .random_forest_classifier_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                )?,
            ),
            Self::LogisticRegression(_) => Self::LogisticRegression(
                smartcore::linear::logistic_regression::LogisticRegression::fit(
                    x,
                    y,
                    LogisticRegressionParameters {
                        solver: settings
                            .logistic_regression_settings
                            .as_ref()
                            .unwrap()
                            .solver
                            .clone(),
                        alpha: INPUT::from(
                            settings
                                .logistic_regression_settings
                                .as_ref()
                                .unwrap()
                                .alpha,
                        )
                        .unwrap(),
                    },
                )?,
            ),
        })
    }

    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> Result<(CrossValidationResult, Self), Failed> {
        match self {
            Self::DecisionTreeClassifier(_) => Self::cross_validate_with(
                self,
                smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::new(),
                settings
                    .decision_tree_classifier_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            Self::KNNClassifier(_) => Self::cross_validate_with(
                self,
                smartcore::neighbors::knn_classifier::KNNClassifier::new(),
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
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            Self::RandomForestClassifier(_) => Self::cross_validate_with(
                self,
                smartcore::ensemble::random_forest_classifier::RandomForestClassifier::new(),
                settings
                    .random_forest_classifier_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            Self::LogisticRegression(_) => Self::cross_validate_with(
                self,
                smartcore::linear::logistic_regression::LogisticRegression::new(),
                LogisticRegressionParameters {
                    solver: settings
                        .logistic_regression_settings
                        .as_ref()
                        .unwrap()
                        .solver
                        .clone(),
                    alpha: INPUT::from(
                        settings
                            .logistic_regression_settings
                            .as_ref()
                            .unwrap()
                            .alpha,
                    )
                    .unwrap(),
                },
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
        }
    }

    fn metric(settings: &ClassificationSettings) -> fn(&OutputArray, &OutputArray) -> f64 {
        settings.get_metric::<OUTPUT, OutputArray>()
    }
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
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
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

    /// Default random forest classifier algorithm
    #[must_use]
    pub fn default_random_forest_classifier() -> Self {
        Self::RandomForestClassifier(
            smartcore::ensemble::random_forest_classifier::RandomForestClassifier::new(),
        )
    }

    /// Default logistic regression classifier algorithm
    #[must_use]
    pub fn default_logistic_regression() -> Self {
        Self::LogisticRegression(smartcore::linear::logistic_regression::LogisticRegression::new())
    }

    /// Get a vector of all possible algorithms
    #[must_use]
    pub fn all_algorithms(settings: &ClassificationSettings) -> Vec<Self> {
        <Self as Algorithm<ClassificationSettings>>::all_algorithms(settings)
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Algorithm<ClassificationSettings>
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
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    type Input = INPUT;
    type Output = OUTPUT;
    type InputArray = InputArray;
    type OutputArray = OutputArray;

    fn predict(&self, x: &Self::InputArray) -> Result<Self::OutputArray, Failed> {
        match self {
            Self::DecisionTreeClassifier(model) => model.predict(x),
            Self::KNNClassifier(model) => model.predict(x),
            Self::RandomForestClassifier(model) => model.predict(x),
            Self::LogisticRegression(model) => model.predict(x),
        }
    }

    fn cross_validate_model(
        self,
        x: &Self::InputArray,
        y: &Self::OutputArray,
        settings: &ClassificationSettings,
    ) -> Result<ComparisonEntry<Self>, Failed> {
        let start = Instant::now();
        let results = self.cv(x, y, settings)?;
        let end = Instant::now();
        Ok(ComparisonEntry {
            result: results.0,
            algorithm: results.1,
            duration: end.duration_since(start),
        })
    }

    fn all_algorithms(settings: &ClassificationSettings) -> Vec<Self> {
        let mut algorithms = vec![Self::default_decision_tree_classifier()];
        if settings.knn_classifier_settings.is_some() {
            algorithms.push(Self::default_knn_classifier());
        }
        if settings.random_forest_classifier_settings.is_some() {
            algorithms.push(Self::default_random_forest_classifier());
        }
        if settings.logistic_regression_settings.is_some() {
            algorithms.push(Self::default_logistic_regression());
        }
        algorithms
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
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn eq(&self, other: &Self) -> bool {
        matches!(self, Self::DecisionTreeClassifier(_))
            && matches!(other, Self::DecisionTreeClassifier(_))
            || matches!(self, Self::KNNClassifier(_)) && matches!(other, Self::KNNClassifier(_))
            || matches!(self, Self::RandomForestClassifier(_))
                && matches!(other, Self::RandomForestClassifier(_))
            || matches!(self, Self::LogisticRegression(_))
                && matches!(other, Self::LogisticRegression(_))
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
        + EVDDecomposable<INPUT>
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
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecisionTreeClassifier(_) => write!(f, "Decision Tree Classifier"),
            Self::KNNClassifier(_) => write!(f, "KNN Classifier"),
            Self::RandomForestClassifier(_) => write!(f, "Random Forest Classifier"),
            Self::LogisticRegression(_) => write!(f, "Logistic Regression"),
        }
    }
}
