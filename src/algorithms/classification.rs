//! Classification algorithm definitions and helpers.

use std::fmt::{Display, Formatter};
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::{ClassificationSettings, SettingsError};
use crate::utils::distance::KNNRegressorDistance;
use num_traits::Unsigned;
use smartcore::api::SupervisedEstimator;
use smartcore::error::{Failed, FailedError};
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::linear::logistic_regression::LogisticRegressionParameters;
use smartcore::model_selection::{BaseKFold, CrossValidationResult};
use smartcore::naive_bayes::{
    categorical::CategoricalNB, multinomial::MultinomialNB as SmartcoreMultinomialNB,
};
use smartcore::numbers::basenum::Number;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

type DenseCategoricalNB<OUTPUT, OutputArray> =
    CategoricalNB<OUTPUT, DenseMatrix<OUTPUT>, OutputArray>;
type DenseMultinomialNB<OUTPUT, OutputArray> =
    SmartcoreMultinomialNB<OUTPUT, OUTPUT, DenseMatrix<OUTPUT>, OutputArray>;

const CATEGORICAL_NB_ALGORITHM_NAME: &str = "categorical naive Bayes";
const MULTINOMIAL_NB_ALGORITHM_NAME: &str = "multinomial naive Bayes";

fn convert_to_nonnegative_integer_dense_matrix<INPUT, OUTPUT, InputArray>(
    x: &InputArray,
    algorithm_name: &'static str,
) -> Result<DenseMatrix<OUTPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
    InputArray: Array2<INPUT>,
{
    let (rows, cols) = x.shape();
    let mut data: Vec<Vec<OUTPUT>> = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut row_values: Vec<OUTPUT> = Vec::with_capacity(cols);
        for col in 0..cols {
            let value = *x.get((row, col));
            row_values.push(convert_feature_value(value, row, col, algorithm_name)?);
        }
        data.push(row_values);
    }
    DenseMatrix::from_2d_vec(&data)
}

fn convert_feature_value<INPUT, OUTPUT>(
    value: INPUT,
    row: usize,
    col: usize,
    algorithm_name: &'static str,
) -> Result<OUTPUT, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
{
    if !value.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!("{algorithm_name} requires finite feature values (row {row}, column {col})"),
        ));
    }

    let as_f64 = value
        .to_f64()
        .ok_or_else(|| {
            Failed::because(
                FailedError::ParametersError,
                &format!(
                    "{algorithm_name} could not convert feature value {value} at row {row}, column {col}"
                ),
            )
        })?;

    if (as_f64 - as_f64.round()).abs() > f64::EPSILON {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!(
                "{algorithm_name} requires integer-valued features (row {row}, column {col}, value {value})"
            ),
        ));
    }

    if as_f64 < 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!(
                "{algorithm_name} requires non-negative feature values (row {row}, column {col}, value {value})"
            ),
        ));
    }

    let rounded = as_f64.round();
    OUTPUT::from_f64(rounded).ok_or_else(|| {
        Failed::because(
            FailedError::ParametersError,
            &format!(
                "{algorithm_name} feature value {value} at row {row}, column {col} exceeds supported range"
            ),
        )
    })
}

fn convert_to_categorical_dense_matrix<INPUT, OUTPUT, InputArray>(
    x: &InputArray,
) -> Result<DenseMatrix<OUTPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
    InputArray: Array2<INPUT>,
{
    convert_to_nonnegative_integer_dense_matrix(x, CATEGORICAL_NB_ALGORITHM_NAME)
}

fn convert_to_multinomial_dense_matrix<INPUT, OUTPUT, InputArray>(
    x: &InputArray,
) -> Result<DenseMatrix<OUTPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
    InputArray: Array2<INPUT>,
{
    convert_to_nonnegative_integer_dense_matrix(x, MULTINOMIAL_NB_ALGORITHM_NAME)
}

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
    /// Categorical naive Bayes classifier
    CategoricalNB(DenseCategoricalNB<OUTPUT, OutputArray>),
    /// Multinomial naive Bayes classifier
    MultinomialNB(DenseMultinomialNB<OUTPUT, OutputArray>),
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
    #[allow(clippy::too_many_lines)]
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
                    .to_classifier_params::<INPUT>()
                    .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
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
            Self::CategoricalNB(_) => {
                let params = settings.categorical_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Categorical NB settings not provided",
                    )
                })?;
                let converted = convert_to_categorical_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                Self::CategoricalNB(CategoricalNB::fit(&converted, y, params)?)
            }
            Self::MultinomialNB(_) => {
                let params = settings.multinomial_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Multinomial NB settings not provided",
                    )
                })?;
                let converted = convert_to_multinomial_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                Self::MultinomialNB(SmartcoreMultinomialNB::fit(&converted, y, params)?)
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
        let metric = Self::metric(settings)
            .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
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
                metric,
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
                    .to_classifier_params::<INPUT>()
                    .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
                Self::cross_validate_with(
                    self,
                    smartcore::neighbors::knn_classifier::KNNClassifier::new(),
                    params,
                    x,
                    y,
                    settings,
                    &settings.get_kfolds(),
                    metric,
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
                metric,
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
                    metric,
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
                metric,
            ),
            Self::CategoricalNB(_) => {
                let params = settings.categorical_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Categorical NB settings not provided",
                    )
                })?;
                let converted = convert_to_categorical_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                let kfold = settings.get_kfolds();
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(&converted) {
                    let train_x = converted.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = converted.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let fold_model = CategoricalNB::fit(&train_x, &train_y, params.clone())?;
                    let train_pred = fold_model.predict(&train_x)?;
                    let test_pred = fold_model.predict(&test_x)?;
                    train_scores.push(metric(&train_y, &train_pred));
                    test_scores.push(metric(&test_y, &test_pred));
                }
                let result = CrossValidationResult {
                    test_score: test_scores,
                    train_score: train_scores,
                };
                let model = CategoricalNB::fit(&converted, y, params)?;
                Ok((result, Self::CategoricalNB(model)))
            }
            Self::MultinomialNB(_) => {
                let params = settings.multinomial_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Multinomial NB settings not provided",
                    )
                })?;
                let converted = convert_to_multinomial_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                let kfold = settings.get_kfolds();
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(&converted) {
                    let train_x = converted.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = converted.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let fold_model =
                        SmartcoreMultinomialNB::fit(&train_x, &train_y, params.clone())?;
                    let train_pred = fold_model.predict(&train_x)?;
                    let test_pred = fold_model.predict(&test_x)?;
                    train_scores.push(metric(&train_y, &train_pred));
                    test_scores.push(metric(&test_y, &test_pred));
                }
                let result = CrossValidationResult {
                    test_score: test_scores,
                    train_score: train_scores,
                };
                let model = SmartcoreMultinomialNB::fit(&converted, y, params)?;
                Ok((result, Self::MultinomialNB(model)))
            }
        }
    }

    fn metric(
        settings: &ClassificationSettings,
    ) -> Result<fn(&OutputArray, &OutputArray) -> f64, SettingsError> {
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

    /// Default categorical naive Bayes classifier algorithm
    #[must_use]
    pub fn default_categorical_nb() -> Self {
        Self::CategoricalNB(DenseCategoricalNB::<OUTPUT, OutputArray>::new())
    }

    /// Default multinomial naive Bayes classifier algorithm
    #[must_use]
    pub fn default_multinomial_nb() -> Self {
        Self::MultinomialNB(DenseMultinomialNB::<OUTPUT, OutputArray>::new())
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
            Self::CategoricalNB(model) => {
                let converted = convert_to_categorical_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                model.predict(&converted)
            }
            Self::MultinomialNB(model) => {
                let converted = convert_to_multinomial_dense_matrix::<INPUT, OUTPUT, _>(x)?;
                model.predict(&converted)
            }
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
        if settings.categorical_nb_settings.is_some() {
            algorithms.push(Self::default_categorical_nb());
        }
        if settings.multinomial_nb_settings.is_some() {
            algorithms.push(Self::default_multinomial_nb());
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
            || matches!(self, Self::CategoricalNB(_)) && matches!(other, Self::CategoricalNB(_))
            || matches!(self, Self::MultinomialNB(_)) && matches!(other, Self::MultinomialNB(_))
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
            Self::CategoricalNB(_) => write!(f, "Categorical NB"),
            Self::MultinomialNB(_) => write!(f, "Multinomial NB"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassificationAlgorithm, ClassificationSettings};
    use crate::DenseMatrix;
    use crate::settings::CategoricalNBParameters;
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

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn categorical_nb_requires_settings() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let mut settings = ClassificationSettings::default();
        settings.categorical_nb_settings = None;
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_categorical_nb();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn multinomial_nb_requires_settings() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let mut settings = ClassificationSettings::default();
        settings.multinomial_nb_settings = None;
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_multinomial_nb();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
    }

    #[test]
    fn categorical_nb_rejects_fractional_features() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.5_f64, 0.0_f64], &[1.0_f64, 1.5_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let settings = ClassificationSettings::default()
            .with_categorical_nb_settings(CategoricalNBParameters::default());
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_categorical_nb();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
        assert!(
            err.to_string()
                .contains("categorical naive Bayes requires integer-valued features")
        );
    }

    #[test]
    fn categorical_nb_trains_on_integer_features() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 1.0_f64], &[1.0_f64, 0.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let settings = ClassificationSettings::default()
            .with_categorical_nb_settings(CategoricalNBParameters::default());
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_categorical_nb();
        assert!(algo.fit(&x, &y, &settings).is_ok());
    }
}
