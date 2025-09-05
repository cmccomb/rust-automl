//! Classification algorithm definitions and helpers.

use std::fmt::{Display, Formatter};
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::ClassificationSettings;
use crate::utils::distance::KNNRegressorDistance;
use num_traits::Unsigned;
use smartcore::api::SupervisedEstimator;
use smartcore::error::{Failed, FailedError};
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::basenum::Number;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

/// Supported classification algorithms.
pub enum ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
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
            KNNRegressorDistance<INPUT>,
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
    /// Gaussian naive Bayes classifier
    GaussianNB(
        smartcore::naive_bayes::gaussian::GaussianNB<INPUT, OUTPUT, InputArray, OutputArray>,
    ),
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    SupervisedTrain<INPUT, OUTPUT, InputArray, OutputArray, ClassificationSettings>
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
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
                        .clone()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "decision tree classifier settings not provided",
                            )
                        })?,
                )?,
            ),
            Self::KNNClassifier(_) => {
                let params = settings
                    .knn_classifier_settings
                    .as_ref()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "KNN classifier settings not provided",
                        )
                    })?
                    .to_classifier_params::<INPUT>();
                Self::KNNClassifier(smartcore::neighbors::knn_classifier::KNNClassifier::fit(
                    x, y, params,
                )?)
            }
            Self::RandomForestClassifier(_) => Self::RandomForestClassifier(
                smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                    x,
                    y,
                    settings
                        .random_forest_classifier_settings
                        .clone()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "random forest classifier settings not provided",
                            )
                        })?,
                )?,
            ),
            Self::LogisticRegression(_) => {
                let lr_settings =
                    settings
                        .logistic_regression_settings
                        .as_ref()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "logistic regression settings not provided",
                            )
                        })?;

                let params = LogisticRegressionParameters {
                    solver: lr_settings.solver.clone(),
                    alpha: {
                        let alpha = INPUT::from(lr_settings.alpha).ok_or_else(|| {
                            Failed::input("alpha value cannot be represented as input type")
                        })?;
                        if !alpha.is_finite() {
                            return Err(Failed::input("alpha value must be finite"));
                        }
                        alpha
                    },
                };

                Self::LogisticRegression(
                    smartcore::linear::logistic_regression::LogisticRegression::fit(x, y, params)?,
                )
            }
            Self::GaussianNB(_) => {
                Self::GaussianNB(smartcore::naive_bayes::gaussian::GaussianNB::fit(
                    x,
                    y,
                    settings.gaussian_nb_settings.clone().ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "Gaussian NB settings not provided",
                        )
                    })?,
                )?)
            }
        })
    }

    #[allow(clippy::too_many_lines)]
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
                    .clone()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "decision tree classifier settings not provided",
                        )
                    })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            Self::KNNClassifier(_) => {
                let params = settings
                    .knn_classifier_settings
                    .as_ref()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "KNN classifier settings not provided",
                        )
                    })?
                    .to_classifier_params::<INPUT>();
                Self::cross_validate_with(
                    self,
                    smartcore::neighbors::knn_classifier::KNNClassifier::new(),
                    params,
                    x,
                    y,
                    settings,
                    &settings.get_kfolds(),
                    Self::metric(settings),
                )
            }
            Self::RandomForestClassifier(_) => Self::cross_validate_with(
                self,
                smartcore::ensemble::random_forest_classifier::RandomForestClassifier::new(),
                settings
                    .random_forest_classifier_settings
                    .clone()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "random forest classifier settings not provided",
                        )
                    })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            Self::LogisticRegression(_) => {
                let lr_settings =
                    settings
                        .logistic_regression_settings
                        .as_ref()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "logistic regression settings not provided",
                            )
                        })?;

                let params = LogisticRegressionParameters {
                    solver: lr_settings.solver.clone(),
                    alpha: {
                        let alpha = INPUT::from(lr_settings.alpha).ok_or_else(|| {
                            Failed::input("alpha value cannot be represented as input type")
                        })?;
                        if !alpha.is_finite() {
                            return Err(Failed::input("alpha value must be finite"));
                        }
                        alpha
                    },
                };

                Self::cross_validate_with(
                    self,
                    smartcore::linear::logistic_regression::LogisticRegression::new(),
                    params,
                    x,
                    y,
                    settings,
                    &settings.get_kfolds(),
                    Self::metric(settings),
                )
            }
            Self::GaussianNB(_) => Self::cross_validate_with(
                self,
                smartcore::naive_bayes::gaussian::GaussianNB::new(),
                settings.gaussian_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Gaussian NB settings not provided",
                    )
                })?,
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
    OUTPUT: Number + Ord + Unsigned,
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

    /// Default Gaussian naive Bayes classifier algorithm
    #[must_use]
    pub fn default_gaussian_nb() -> Self {
        Self::GaussianNB(smartcore::naive_bayes::gaussian::GaussianNB::new())
    }

    /// Get a vector of all possible algorithms
    #[must_use]
    pub fn all_algorithms(settings: &ClassificationSettings) -> Vec<Self> {
        <Self as Algorithm<ClassificationSettings>>::all_algorithms(settings)
    }

    /// Fit the algorithm using the provided settings.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] if training is not successful.
    #[allow(clippy::missing_errors_doc)]
    pub fn fit(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> Result<Self, Failed> {
        <Self as SupervisedTrain<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            ClassificationSettings,
        >>::fit(self, x, y, settings)
    }

    /// Perform cross-validation for the algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] if cross-validation fails.
    #[allow(clippy::missing_errors_doc)]
    pub fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &ClassificationSettings,
    ) -> Result<(CrossValidationResult, Self), Failed> {
        <Self as SupervisedTrain<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            ClassificationSettings,
        >>::cv(self, x, y, settings)
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Algorithm<ClassificationSettings>
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
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
            Self::GaussianNB(model) => model.predict(x),
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
        if settings.gaussian_nb_settings.is_some() {
            algorithms.push(Self::default_gaussian_nb());
        }
        algorithms
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> PartialEq
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
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
            || matches!(self, Self::GaussianNB(_)) && matches!(other, Self::GaussianNB(_))
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
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
    OUTPUT: Number + Ord + Unsigned,
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
            Self::GaussianNB(_) => write!(f, "Gaussian NB"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassificationAlgorithm, ClassificationSettings};
    use crate::DenseMatrix;
    use smartcore::error::FailedError;

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn logistic_regression_requires_settings() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let mut settings = ClassificationSettings::default();
        settings.logistic_regression_settings = None;
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_logistic_regression();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn gaussian_nb_requires_settings() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let mut settings = ClassificationSettings::default();
        settings.gaussian_nb_settings = None;
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_gaussian_nb();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
    }
}
