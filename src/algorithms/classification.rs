//! Classification algorithm definitions and helpers.

use std::fmt::{Display, Formatter};
use std::mem;
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::{
    BernoulliNBParameters, ClassificationSettings, SVCParameters, SettingsError,
};
use crate::utils::distance::KNNRegressorDistance;
use crate::utils::kernels::SmartcoreKernel;
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
    bernoulli::{
        BernoulliNB as SmartcoreBernoulliNB,
        BernoulliNBParameters as SmartcoreBernoulliNBParameters,
    },
    categorical::CategoricalNB,
    multinomial::MultinomialNB as SmartcoreMultinomialNB,
};
use smartcore::numbers::basenum::Number;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;
use smartcore::svm::svc::{MultiClassSVC, SVCParameters as SmartcoreSVCParameters};

type DenseCategoricalNB<OUTPUT, OutputArray> =
    CategoricalNB<OUTPUT, DenseMatrix<OUTPUT>, OutputArray>;
type DenseMultinomialNB<OUTPUT, OutputArray> =
    SmartcoreMultinomialNB<OUTPUT, OUTPUT, DenseMatrix<OUTPUT>, OutputArray>;
type DenseBernoulliNB<INPUT, OUTPUT, OutputArray> =
    SmartcoreBernoulliNB<INPUT, OUTPUT, DenseMatrix<INPUT>, OutputArray>;

const CATEGORICAL_NB_ALGORITHM_NAME: &str = "categorical naive Bayes";
const MULTINOMIAL_NB_ALGORITHM_NAME: &str = "multinomial naive Bayes";
const BERNOULLI_NB_ALGORITHM_NAME: &str = "Bernoulli naive Bayes";

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

fn prepare_bernoulli_dense_matrix<INPUT, InputArray>(
    x: &InputArray,
    threshold: Option<INPUT>,
) -> Result<DenseMatrix<INPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Array2<INPUT>,
{
    let (rows, cols) = x.shape();
    let mut data: Vec<Vec<INPUT>> = Vec::with_capacity(rows);
    if let Some(threshold_value) = threshold {
        for row in 0..rows {
            let mut row_values: Vec<INPUT> = Vec::with_capacity(cols);
            for col in 0..cols {
                let value = *x.get((row, col));
                if !value.is_finite() {
                    return Err(Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "{BERNOULLI_NB_ALGORITHM_NAME} requires finite feature values (row {row}, column {col})"
                        ),
                    ));
                }
                if value > threshold_value {
                    row_values.push(INPUT::one());
                } else {
                    row_values.push(INPUT::zero());
                }
            }
            data.push(row_values);
        }
    } else {
        let zero = INPUT::zero();
        let one = INPUT::one();
        for row in 0..rows {
            let mut row_values: Vec<INPUT> = Vec::with_capacity(cols);
            for col in 0..cols {
                let value = *x.get((row, col));
                if !value.is_finite() {
                    return Err(Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "{BERNOULLI_NB_ALGORITHM_NAME} requires finite feature values (row {row}, column {col})"
                        ),
                    ));
                }
                if value == zero {
                    row_values.push(zero);
                } else if value == one {
                    row_values.push(one);
                } else {
                    return Err(Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "{BERNOULLI_NB_ALGORITHM_NAME} requires binary features (row {row}, column {col}, value {value})"
                        ),
                    ));
                }
            }
            data.push(row_values);
        }
    }

    DenseMatrix::from_2d_vec(&data).map_err(|err| {
        Failed::because(
            FailedError::ParametersError,
            &format!(
                "{BERNOULLI_NB_ALGORITHM_NAME} could not construct a binary feature matrix: {err}"
            ),
        )
    })
}

fn convert_bernoulli_parameters<INPUT>(
    params: &BernoulliNBParameters<f64>,
) -> Result<SmartcoreBernoulliNBParameters<INPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
{
    let binarize = match params.binarize {
        Some(value) => {
            if !value.is_finite() {
                return Err(Failed::because(
                    FailedError::ParametersError,
                    &format!(
                        "{BERNOULLI_NB_ALGORITHM_NAME} binarize threshold must be finite (value {value})"
                    ),
                ));
            }
            Some(
                INPUT::from_f64(value).ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "{BERNOULLI_NB_ALGORITHM_NAME} binarize threshold {value} cannot be represented by the input type"
                        ),
                    )
                })?,
            )
        }
        None => None,
    };

    Ok(SmartcoreBernoulliNBParameters {
        alpha: params.alpha,
        priors: params.priors.clone(),
        binarize,
    })
}

#[derive(Clone)]
struct PreparedSVCParameters<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    epoch: usize,
    c: INPUT,
    tol: INPUT,
    kernel_template: smartcore::svm::Kernels,
}

impl<INPUT> PreparedSVCParameters<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn new(settings: &SVCParameters) -> Result<(Self, Box<dyn smartcore::svm::Kernel>), Failed> {
        let SmartcoreKernel { kernel, function } = settings.kernel.to_smartcore()?;
        let c = convert_positive_scalar::<INPUT>(settings.c, "support vector classifier C")?;
        let tol =
            convert_positive_scalar::<INPUT>(settings.tol, "support vector classifier tolerance")?;
        Ok((
            Self {
                epoch: settings.epoch,
                c,
                tol,
                kernel_template: kernel,
            },
            function,
        ))
    }

    fn boxed_params<OUTPUT, InputArray, OutputArray>(
        &self,
        kernel: Box<dyn smartcore::svm::Kernel>,
    ) -> Box<SmartcoreSVCParameters<INPUT, OUTPUT, InputArray, OutputArray>>
    where
        OUTPUT: Number + Ord,
        InputArray: Array2<INPUT>,
        OutputArray: Array1<OUTPUT>,
    {
        let mut params = SmartcoreSVCParameters::default();
        params.epoch = self.epoch;
        params.c = self.c;
        params.tol = self.tol;
        params.kernel = Some(kernel);
        Box::new(params)
    }

    fn boxed_params_from_template<OUTPUT, InputArray, OutputArray>(
        &self,
    ) -> Box<SmartcoreSVCParameters<INPUT, OUTPUT, InputArray, OutputArray>>
    where
        OUTPUT: Number + Ord,
        InputArray: Array2<INPUT>,
        OutputArray: Array1<OUTPUT>,
    {
        self.boxed_params::<OUTPUT, InputArray, OutputArray>(Box::new(self.kernel_template.clone()))
    }
}

/// Support vector classifier wrapper holding owned kernel parameters.
pub struct OwnedSupportVectorClassifier<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
{
    /// Stored parameters ensure the kernel object outlives the trained model.
    _parameters: Box<SmartcoreSVCParameters<INPUT, OUTPUT, InputArray, OutputArray>>,
    model: MultiClassSVC<'static, INPUT, OUTPUT, InputArray, OutputArray>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    OwnedSupportVectorClassifier<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
{
    fn fit_with_parameters(
        x: &InputArray,
        y: &OutputArray,
        params: Box<SmartcoreSVCParameters<INPUT, OUTPUT, InputArray, OutputArray>>,
    ) -> Result<Self, Failed> {
        let params_ref: &SmartcoreSVCParameters<INPUT, OUTPUT, InputArray, OutputArray> =
            params.as_ref();
        let model = MultiClassSVC::fit(x, y, params_ref)?;
        let model = unsafe {
            mem::transmute::<
                MultiClassSVC<'_, INPUT, OUTPUT, InputArray, OutputArray>,
                MultiClassSVC<'static, INPUT, OUTPUT, InputArray, OutputArray>,
            >(model)
        };
        Ok(Self {
            _parameters: params,
            model,
        })
    }

    fn predict_vec(&self, x: &InputArray) -> Result<Vec<OUTPUT>, Failed> {
        let raw_predictions = self.model.predict(x)?;
        raw_predictions
            .into_iter()
            .map(convert_svc_prediction::<INPUT, OUTPUT>)
            .collect()
    }

    fn predict_array(&self, x: &InputArray) -> Result<OutputArray, Failed> {
        let values = self.predict_vec(x)?;
        Ok(<OutputArray as Array1<OUTPUT>>::from_vec_slice(&values))
    }
}

fn convert_positive_scalar<INPUT>(value: f32, name: &str) -> Result<INPUT, Failed>
where
    INPUT: RealNumber + FloatNumber,
{
    let as_f64 = f64::from(value);
    if !as_f64.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!("{name} must be finite"),
        ));
    }
    if as_f64 <= 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!("{name} must be positive"),
        ));
    }
    INPUT::from_f64(as_f64).ok_or_else(|| {
        Failed::because(
            FailedError::ParametersError,
            &format!("{name} value {as_f64} cannot be represented by the input type"),
        )
    })
}

fn convert_svc_prediction<INPUT, OUTPUT>(value: INPUT) -> Result<OUTPUT, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
{
    let as_f64 = value
        .to_f64()
        .ok_or_else(|| Failed::predict("prediction not representable"))?;
    if !as_f64.is_finite() {
        return Err(Failed::predict(
            "support vector classifier produced a non-finite prediction",
        ));
    }
    let rounded = as_f64.round();
    if (rounded - as_f64).abs() > f64::EPSILON {
        return Err(Failed::predict(&format!(
            "support vector classifier produced a non-integer class value {as_f64}"
        )));
    }
    OUTPUT::from_f64(rounded).ok_or_else(|| {
        Failed::predict(
            &format!(
                "support vector classifier prediction {rounded} cannot be represented in the output type"
            ),
        )
    })
}

/// Wrapper around Smartcore's Bernoulli naive Bayes implementation that stores the
/// optional binarization threshold used during training.
///
/// # Examples
/// ```ignore
/// use automl::algorithms::classification::BernoulliNBModel;
/// let model = BernoulliNBModel::<f64, u32, Vec<u32>>::untrained();
/// assert!(matches!(model.binarize(), None));
/// ```
pub struct BernoulliNBModel<INPUT, OUTPUT, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
    OutputArray: Array1<OUTPUT>,
{
    model: DenseBernoulliNB<INPUT, OUTPUT, OutputArray>,
    binarize: Option<INPUT>,
}

impl<INPUT, OUTPUT, OutputArray> BernoulliNBModel<INPUT, OUTPUT, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord + Unsigned,
    OutputArray: Array1<OUTPUT>,
{
    fn untrained() -> Self {
        Self {
            model: DenseBernoulliNB::<INPUT, OUTPUT, OutputArray>::new(),
            binarize: None,
        }
    }

    fn trained(
        model: DenseBernoulliNB<INPUT, OUTPUT, OutputArray>,
        binarize: Option<INPUT>,
    ) -> Self {
        Self { model, binarize }
    }

    /// Return the binarization threshold recorded during training.
    #[must_use]
    pub fn binarize(&self) -> Option<INPUT> {
        self.binarize
    }

    fn predict<InputArray>(&self, x: &InputArray) -> Result<OutputArray, Failed>
    where
        InputArray: Array2<INPUT>,
    {
        let converted = prepare_bernoulli_dense_matrix(x, self.binarize)?;
        self.model.predict(&converted)
    }
}

/// Supported classification algorithms.
pub enum ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
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
    /// Support vector classifier
    SupportVectorClassifier(
        Option<OwnedSupportVectorClassifier<INPUT, OUTPUT, InputArray, OutputArray>>,
    ),
    /// Bernoulli naive Bayes classifier
    BernoulliNB(BernoulliNBModel<INPUT, OUTPUT, OutputArray>),
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
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
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
            Self::SupportVectorClassifier(_) => {
                let svc_settings = settings.svc_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "support vector classifier settings not provided",
                    )
                })?;
                let (prepared, initial_kernel) = PreparedSVCParameters::<INPUT>::new(svc_settings)?;
                let params =
                    prepared.boxed_params::<OUTPUT, InputArray, OutputArray>(initial_kernel);
                let model = OwnedSupportVectorClassifier::fit_with_parameters(x, y, params)?;
                Self::SupportVectorClassifier(Some(model))
            }
            Self::BernoulliNB(_) => {
                let params = settings.bernoulli_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Bernoulli NB settings not provided",
                    )
                })?;
                let typed_params = convert_bernoulli_parameters::<INPUT>(&params)?;
                let threshold = typed_params.binarize;
                let converted = prepare_bernoulli_dense_matrix::<INPUT, _>(x, threshold)?;
                let mut fit_params = typed_params.clone();
                fit_params.binarize = None;
                let model = SmartcoreBernoulliNB::fit(&converted, y, fit_params)?;
                Self::BernoulliNB(BernoulliNBModel::trained(model, threshold))
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
            Self::SupportVectorClassifier(_) => {
                let svc_settings = settings.svc_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "support vector classifier settings not provided",
                    )
                })?;
                let (prepared, initial_kernel) = PreparedSVCParameters::<INPUT>::new(svc_settings)?;
                let kfold = settings.get_kfolds();
                let mut initial_kernel = Some(initial_kernel);
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(x) {
                    let train_x = x.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = x.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let params = if let Some(kernel) = initial_kernel.take() {
                        prepared.boxed_params::<OUTPUT, InputArray, OutputArray>(kernel)
                    } else {
                        prepared.boxed_params_from_template::<OUTPUT, InputArray, OutputArray>()
                    };
                    let fold_model = OwnedSupportVectorClassifier::fit_with_parameters(
                        &train_x, &train_y, params,
                    )?;
                    let train_pred = fold_model.predict_array(&train_x)?;
                    let test_pred = fold_model.predict_array(&test_x)?;
                    train_scores.push(metric(&train_y, &train_pred));
                    test_scores.push(metric(&test_y, &test_pred));
                }
                let result = CrossValidationResult {
                    test_score: test_scores,
                    train_score: train_scores,
                };
                let (final_prepared, final_kernel) =
                    PreparedSVCParameters::<INPUT>::new(svc_settings)?;
                let final_params =
                    final_prepared.boxed_params::<OUTPUT, InputArray, OutputArray>(final_kernel);
                let final_model =
                    OwnedSupportVectorClassifier::fit_with_parameters(x, y, final_params)?;
                Ok((result, Self::SupportVectorClassifier(Some(final_model))))
            }
            Self::BernoulliNB(_) => {
                let params = settings.bernoulli_nb_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "Bernoulli NB settings not provided",
                    )
                })?;
                let typed_params = convert_bernoulli_parameters::<INPUT>(&params)?;
                let threshold = typed_params.binarize;
                let converted = prepare_bernoulli_dense_matrix::<INPUT, _>(x, threshold)?;
                let mut params_for_fit = typed_params.clone();
                params_for_fit.binarize = None;
                let kfold = settings.get_kfolds();
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(&converted) {
                    let train_x = converted.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = converted.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let fold_model =
                        SmartcoreBernoulliNB::fit(&train_x, &train_y, params_for_fit.clone())?;
                    let train_pred = fold_model.predict(&train_x)?;
                    let test_pred = fold_model.predict(&test_x)?;
                    train_scores.push(metric(&train_y, &train_pred));
                    test_scores.push(metric(&test_y, &test_pred));
                }
                let result = CrossValidationResult {
                    test_score: test_scores,
                    train_score: train_scores,
                };
                let model = SmartcoreBernoulliNB::fit(&converted, y, params_for_fit)?;
                Ok((
                    result,
                    Self::BernoulliNB(BernoulliNBModel::trained(model, threshold)),
                ))
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
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
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

    /// Default support vector classifier algorithm
    #[must_use]
    pub fn default_support_vector_classifier() -> Self {
        Self::SupportVectorClassifier(None)
    }

    /// Default Bernoulli naive Bayes classifier algorithm
    #[must_use]
    pub fn default_bernoulli_nb() -> Self {
        Self::BernoulliNB(BernoulliNBModel::untrained())
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
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
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
            Self::SupportVectorClassifier(model) => {
                let model = model
                    .as_ref()
                    .ok_or_else(|| Failed::predict("support vector classifier is not trained"))?;
                model.predict_array(x)
            }
            Self::BernoulliNB(model) => model.predict(x),
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
        if settings.svc_settings.is_some() {
            algorithms.push(Self::default_support_vector_classifier());
        }
        if settings.bernoulli_nb_settings.is_some() {
            algorithms.push(Self::default_bernoulli_nb());
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
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        matches!(self, Self::DecisionTreeClassifier(_))
            && matches!(other, Self::DecisionTreeClassifier(_))
            || matches!(self, Self::KNNClassifier(_)) && matches!(other, Self::KNNClassifier(_))
            || matches!(self, Self::RandomForestClassifier(_))
                && matches!(other, Self::RandomForestClassifier(_))
            || matches!(self, Self::LogisticRegression(_))
                && matches!(other, Self::LogisticRegression(_))
            || matches!(self, Self::SupportVectorClassifier(_))
                && matches!(other, Self::SupportVectorClassifier(_))
            || matches!(self, Self::BernoulliNB(_)) && matches!(other, Self::BernoulliNB(_))
            || matches!(self, Self::GaussianNB(_)) && matches!(other, Self::GaussianNB(_))
            || matches!(self, Self::CategoricalNB(_)) && matches!(other, Self::CategoricalNB(_))
            || matches!(self, Self::MultinomialNB(_)) && matches!(other, Self::MultinomialNB(_))
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
{
    fn default() -> Self {
        Self::default_decision_tree_classifier()
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: Number + Ord + Unsigned + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT> + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecisionTreeClassifier(_) => write!(f, "Decision Tree Classifier"),
            Self::KNNClassifier(_) => write!(f, "KNN Classifier"),
            Self::RandomForestClassifier(_) => write!(f, "Random Forest Classifier"),
            Self::LogisticRegression(_) => write!(f, "Logistic Regression"),
            Self::SupportVectorClassifier(_) => write!(f, "Support Vector Classifier"),
            Self::BernoulliNB(_) => write!(f, "Bernoulli NB"),
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
    fn bernoulli_nb_requires_settings() {
        let x: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[0.0_f64, 0.0_f64], &[1.0_f64, 1.0_f64]]).unwrap();
        let y: Vec<u32> = vec![0, 1];
        let mut settings = ClassificationSettings::default();
        settings.bernoulli_nb_settings = None;
        let algo: ClassificationAlgorithm<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ClassificationAlgorithm::default_bernoulli_nb();
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
