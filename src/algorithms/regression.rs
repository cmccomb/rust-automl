//!
//! `RegressionAlgorithm` definitions and helpers

use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::{RegressionSettings, SVRParameters, SettingsError, XGRegressorParameters};
use crate::utils::distance::{Distance, KNNRegressorDistance};
use crate::utils::kernels::SmartcoreKernel;
use smartcore::api::SupervisedEstimator;
use smartcore::error::{Failed, FailedError};
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::model_selection::{BaseKFold, CrossValidationResult};
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;
use smartcore::svm::svr::{SVR as SmartcoreSVR, SVRParameters as SmartcoreSVRParameters};
use smartcore::xgboost::xgb_regressor::{
    XGRegressor as SmartcoreXGRegressor, XGRegressorParameters as SmartcoreXGRegressorParameters,
};

#[derive(Clone)]
struct PreparedSVRParameters<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    eps: INPUT,
    c: INPUT,
    tol: INPUT,
    kernel_template: smartcore::svm::Kernels,
}

impl<INPUT> PreparedSVRParameters<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn new(settings: &SVRParameters) -> Result<Self, Failed> {
        let SmartcoreKernel { kernel, .. } = settings.kernel.to_smartcore()?;
        let eps =
            convert_nonnegative_scalar::<INPUT>(settings.eps, "support vector regressor epsilon")?;
        let c = convert_positive_scalar::<INPUT>(settings.c, "support vector regressor C")?;
        let tol =
            convert_positive_scalar::<INPUT>(settings.tol, "support vector regressor tolerance")?;
        Ok(Self {
            eps,
            c,
            tol,
            kernel_template: kernel,
        })
    }

    fn to_parameters(&self) -> SmartcoreSVRParameters<INPUT> {
        SmartcoreSVRParameters {
            eps: self.eps,
            c: self.c,
            tol: self.tol,
            kernel: Some(self.kernel_template.clone()),
        }
    }
}

/// Support vector regressor wrapper holding owned kernel parameters.
pub struct OwnedSupportVectorRegressor<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
    _parameters: Box<SmartcoreSVRParameters<INPUT>>,
    model: SmartcoreSVR<'static, INPUT, InputArray, Vec<INPUT>>,
    _marker: PhantomData<(OUTPUT, OutputArray)>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    OwnedSupportVectorRegressor<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
        targets: &Vec<INPUT>,
        params: SmartcoreSVRParameters<INPUT>,
    ) -> Result<Self, Failed> {
        let boxed_params = Box::new(params);
        let params_ref: &SmartcoreSVRParameters<INPUT> = boxed_params.as_ref();
        let model = SmartcoreSVR::fit(x, targets, params_ref)?;
        let model = unsafe {
            mem::transmute::<
                SmartcoreSVR<'_, INPUT, InputArray, Vec<INPUT>>,
                SmartcoreSVR<'static, INPUT, InputArray, Vec<INPUT>>,
            >(model)
        };
        Ok(Self {
            _parameters: boxed_params,
            model,
            _marker: PhantomData,
        })
    }

    fn predict_array(&self, x: &InputArray) -> Result<OutputArray, Failed> {
        let predictions = self.model.predict(x)?;
        convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(predictions)
    }
}

fn convert_nonnegative_scalar<INPUT>(value: f32, name: &str) -> Result<INPUT, Failed>
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
    if as_f64 < 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            &format!("{name} must be non-negative"),
        ));
    }
    INPUT::from_f64(as_f64).ok_or_else(|| {
        Failed::because(
            FailedError::ParametersError,
            &format!("{name} value {as_f64} cannot be represented by the input type"),
        )
    })
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

fn convert_targets_to_input<INPUT, OUTPUT, OutputArray>(
    targets: &OutputArray,
) -> Result<Vec<INPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    OutputArray: Array1<OUTPUT>,
{
    let mut converted = Vec::with_capacity(targets.shape());
    for value in targets.iterator(0) {
        converted.push(convert_output_value_to_input::<INPUT, OUTPUT>(*value)?);
    }
    Ok(converted)
}

fn convert_output_value_to_input<INPUT, OUTPUT>(value: OUTPUT) -> Result<INPUT, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
{
    let as_f64 = value.to_f64().ok_or_else(|| {
        Failed::because(
            FailedError::ParametersError,
            "target value not representable as f64",
        )
    })?;
    if !as_f64.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "target value must be finite",
        ));
    }
    INPUT::from_f64(as_f64).ok_or_else(|| {
        Failed::because(
            FailedError::ParametersError,
            &format!(
                "support vector regressor target {as_f64} cannot be represented by the input type"
            ),
        )
    })
}

fn convert_input_predictions_to_output_array<INPUT, OUTPUT, OutputArray>(
    predictions: Vec<INPUT>,
) -> Result<OutputArray, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    OutputArray: Array1<OUTPUT>,
{
    let converted = convert_input_predictions_to_output_vec::<INPUT, OUTPUT>(predictions)?;
    Ok(<OutputArray as Array1<OUTPUT>>::from_vec_slice(&converted))
}

fn convert_input_predictions_to_output_vec<INPUT, OUTPUT>(
    predictions: Vec<INPUT>,
) -> Result<Vec<OUTPUT>, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
{
    let mut converted = Vec::with_capacity(predictions.len());
    for value in predictions {
        converted.push(convert_input_value_to_output::<INPUT, OUTPUT>(value)?);
    }
    Ok(converted)
}

fn convert_input_value_to_output<INPUT, OUTPUT>(value: INPUT) -> Result<OUTPUT, Failed>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
{
    let as_f64 = value
        .to_f64()
        .ok_or_else(|| Failed::predict("prediction value not representable as f64"))?;
    if !as_f64.is_finite() {
        return Err(Failed::predict(
            "support vector regressor produced a non-finite prediction",
        ));
    }
    OUTPUT::from_f64(as_f64).ok_or_else(|| {
        Failed::predict(&format!(
            "support vector regressor prediction {as_f64} cannot be represented in the output type"
        ))
    })
}

fn sanitize_xgboost_parameters(
    params: &XGRegressorParameters,
) -> Result<SmartcoreXGRegressorParameters, Failed> {
    let sanitized: SmartcoreXGRegressorParameters = params.clone();

    if sanitized.n_estimators == 0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost number of estimators must be positive",
        ));
    }

    if sanitized.max_depth == 0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost maximum depth must be positive",
        ));
    }

    if !sanitized.learning_rate.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost learning rate must be finite",
        ));
    }
    if sanitized.learning_rate <= 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost learning rate must be greater than zero",
        ));
    }

    if sanitized.min_child_weight == 0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost minimum child weight must be positive",
        ));
    }

    if !sanitized.lambda.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost lambda must be finite",
        ));
    }
    if sanitized.lambda < 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost lambda must be non-negative",
        ));
    }

    if !sanitized.gamma.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost gamma must be finite",
        ));
    }
    if sanitized.gamma < 0.0 {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost gamma must be non-negative",
        ));
    }

    if !sanitized.base_score.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost base score must be finite",
        ));
    }

    if !sanitized.subsample.is_finite() {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost subsample ratio must be finite",
        ));
    }
    if !(0.0 < sanitized.subsample && sanitized.subsample <= 1.0) {
        return Err(Failed::because(
            FailedError::ParametersError,
            "xgboost subsample ratio must be in (0, 1]",
        ));
    }

    Ok(sanitized)
}

/// `RegressionAlgorithm` options
pub enum RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
    /// Decision tree regressor
    DecisionTreeRegressor(
        smartcore::tree::decision_tree_regressor::DecisionTreeRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Random forest regressor
    RandomForestRegressor(
        smartcore::ensemble::random_forest_regressor::RandomForestRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Extra trees regressor
    ExtraTreesRegressor(
        smartcore::ensemble::extra_trees_regressor::ExtraTreesRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Linear regressor
    Linear(
        smartcore::linear::linear_regression::LinearRegression<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Ridge regressor
    Ridge(
        smartcore::linear::ridge_regression::RidgeRegression<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Lasso regressor
    Lasso(smartcore::linear::lasso::Lasso<INPUT, OUTPUT, InputArray, OutputArray>),
    /// Elastic net regressor
    ElasticNet(smartcore::linear::elastic_net::ElasticNet<INPUT, OUTPUT, InputArray, OutputArray>),
    /// K-nearest neighbors regressor
    KNNRegressor(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            KNNRegressorDistance<INPUT>,
        >,
    ),
    /// Support vector regressor
    SupportVectorRegressor(
        Option<OwnedSupportVectorRegressor<INPUT, OUTPUT, InputArray, OutputArray>>,
    ),
    /// Gradient boosting regressor (`XGBoost`)
    XGBoostRegressor(Option<SmartcoreXGRegressor<INPUT, OUTPUT, InputArray, OutputArray>>),
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    SupervisedTrain<
        INPUT,
        OUTPUT,
        InputArray,
        OutputArray,
        RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    > for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<Self, Failed> {
        Ok(match self {
            Self::Linear(_) => {
                Self::Linear(smartcore::linear::linear_regression::LinearRegression::fit(
                    x,
                    y,
                    settings.linear_settings.clone().ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "linear regression settings not provided",
                        )
                    })?,
                )?)
            }
            Self::Lasso(_) => Self::Lasso(smartcore::linear::lasso::Lasso::fit(
                x,
                y,
                settings.lasso_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "lasso regression settings not provided",
                    )
                })?,
            )?),
            Self::Ridge(_) => {
                Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::fit(
                    x,
                    y,
                    settings.ridge_settings.clone().ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "ridge regression settings not provided",
                        )
                    })?,
                )?)
            }
            Self::ElasticNet(_) => {
                Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::fit(
                    x,
                    y,
                    settings.elastic_net_settings.clone().ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "elastic net regression settings not provided",
                        )
                    })?,
                )?)
            }
            Self::RandomForestRegressor(_) => Self::RandomForestRegressor(
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::fit(
                    x,
                    y,
                    settings
                        .random_forest_regressor_settings
                        .clone()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "random forest regressor settings not provided",
                            )
                        })?,
                )?,
            ),
            Self::ExtraTreesRegressor(_) => Self::ExtraTreesRegressor(
                smartcore::ensemble::extra_trees_regressor::ExtraTreesRegressor::fit(
                    x,
                    y,
                    settings.extra_trees_settings.clone().ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "extra trees regressor settings not provided",
                        )
                    })?,
                )?,
            ),
            Self::DecisionTreeRegressor(_) => Self::DecisionTreeRegressor(
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
                    x,
                    y,
                    settings
                        .decision_tree_regressor_settings
                        .clone()
                        .ok_or_else(|| {
                            Failed::because(
                                FailedError::ParametersError,
                                "decision tree regressor settings not provided",
                            )
                        })?,
                )?,
            ),
            Self::KNNRegressor(_) => {
                let knn_settings = settings.knn_regressor_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "KNN regressor settings not provided",
                    )
                })?;
                let params = knn_settings
                    .to_regressor_params::<INPUT>()
                    .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
                Self::KNNRegressor(smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    x, y, params,
                )?)
            }
            Self::SupportVectorRegressor(_) => {
                let svr_settings = settings.svr_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "support vector regressor settings not provided",
                    )
                })?;
                let prepared = PreparedSVRParameters::<INPUT>::new(svr_settings)?;
                let params = prepared.to_parameters();
                let targets = convert_targets_to_input::<INPUT, OUTPUT, OutputArray>(y)?;
                let model = OwnedSupportVectorRegressor::fit_with_parameters(x, &targets, params)?;
                Self::SupportVectorRegressor(Some(model))
            }
            Self::XGBoostRegressor(_) => {
                let params = settings.xgboost_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "xgboost regressor settings not provided",
                    )
                })?;
                let sanitized = sanitize_xgboost_parameters(params)?;
                let model = SmartcoreXGRegressor::fit(x, y, sanitized)?;
                Self::XGBoostRegressor(Some(model))
            }
        })
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::type_complexity)]
    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<(CrossValidationResult, Self), Failed> {
        let metric = Self::metric(settings)
            .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
        match self {
            RegressionAlgorithm::Linear(_) => Self::cross_validate_with(
                self,
                smartcore::linear::linear_regression::LinearRegression::new(),
                settings.linear_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "linear regression settings not provided",
                    )
                })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::Ridge(_) => Self::cross_validate_with(
                self,
                smartcore::linear::ridge_regression::RidgeRegression::new(),
                settings.ridge_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "ridge regression settings not provided",
                    )
                })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::Lasso(_) => Self::cross_validate_with(
                self,
                smartcore::linear::lasso::Lasso::new(),
                settings.lasso_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "lasso regression settings not provided",
                    )
                })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::ElasticNet(_) => Self::cross_validate_with(
                self,
                smartcore::linear::elastic_net::ElasticNet::new(),
                settings.elastic_net_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "elastic net regression settings not provided",
                    )
                })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::RandomForestRegressor(_) => Self::cross_validate_with(
                self,
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
                settings
                    .random_forest_regressor_settings
                    .clone()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "random forest regressor settings not provided",
                        )
                    })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::ExtraTreesRegressor(_) => Self::cross_validate_with(
                self,
                smartcore::ensemble::extra_trees_regressor::ExtraTreesRegressor::new(),
                settings.extra_trees_settings.clone().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "extra trees regressor settings not provided",
                    )
                })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::DecisionTreeRegressor(_) => Self::cross_validate_with(
                self,
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
                settings
                    .decision_tree_regressor_settings
                    .clone()
                    .ok_or_else(|| {
                        Failed::because(
                            FailedError::ParametersError,
                            "decision tree regressor settings not provided",
                        )
                    })?,
                x,
                y,
                settings,
                &settings.get_kfolds(),
                metric,
            ),
            RegressionAlgorithm::KNNRegressor(_) => {
                let knn_settings = settings.knn_regressor_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "KNN regressor settings not provided",
                    )
                })?;
                let params = knn_settings
                    .to_regressor_params::<INPUT>()
                    .map_err(|e| Failed::because(FailedError::ParametersError, &e.to_string()))?;
                Self::cross_validate_with(
                    self,
                    smartcore::neighbors::knn_regressor::KNNRegressor::new(),
                    params,
                    x,
                    y,
                    settings,
                    &settings.get_kfolds(),
                    metric,
                )
            }
            RegressionAlgorithm::SupportVectorRegressor(_) => {
                let svr_settings = settings.svr_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "support vector regressor settings not provided",
                    )
                })?;
                let prepared = PreparedSVRParameters::<INPUT>::new(svr_settings)?;
                let kfold = settings.get_kfolds();
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(x) {
                    let train_x = x.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = x.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let train_targets =
                        convert_targets_to_input::<INPUT, OUTPUT, OutputArray>(&train_y)?;
                    let params = prepared.to_parameters();
                    let fold_model = OwnedSupportVectorRegressor::fit_with_parameters(
                        &train_x,
                        &train_targets,
                        params,
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
                let final_params = prepared.to_parameters();
                let final_targets = convert_targets_to_input::<INPUT, OUTPUT, OutputArray>(y)?;
                let final_model = OwnedSupportVectorRegressor::fit_with_parameters(
                    x,
                    &final_targets,
                    final_params,
                )?;
                Ok((result, Self::SupportVectorRegressor(Some(final_model))))
            }
            RegressionAlgorithm::XGBoostRegressor(_) => {
                let params = settings.xgboost_settings.as_ref().ok_or_else(|| {
                    Failed::because(
                        FailedError::ParametersError,
                        "xgboost regressor settings not provided",
                    )
                })?;
                let sanitized = sanitize_xgboost_parameters(params)?;
                let kfold = settings.get_kfolds();
                let mut test_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                let mut train_scores: Vec<f64> = Vec::with_capacity(kfold.n_splits);
                for (train_idx, test_idx) in kfold.split(x) {
                    let train_x = x.take(&train_idx, 0);
                    let train_y = y.take(&train_idx);
                    let test_x = x.take(&test_idx, 0);
                    let test_y = y.take(&test_idx);
                    let fold_model =
                        SmartcoreXGRegressor::fit(&train_x, &train_y, sanitized.clone())?;
                    let train_pred =
                        convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(
                            fold_model.predict(&train_x)?,
                        )?;
                    let test_pred =
                        convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(
                            fold_model.predict(&test_x)?,
                        )?;
                    train_scores.push(metric(&train_y, &train_pred));
                    test_scores.push(metric(&test_y, &test_pred));
                }
                let result = CrossValidationResult {
                    test_score: test_scores,
                    train_score: train_scores,
                };
                let final_model = SmartcoreXGRegressor::fit(x, y, sanitized)?;
                Ok((result, Self::XGBoostRegressor(Some(final_model))))
            }
        }
    }

    fn metric(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<fn(&OutputArray, &OutputArray) -> f64, SettingsError> {
        settings.get_metric()
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
    /// Default linear regression algorithm
    #[must_use]
    pub fn default_linear() -> Self {
        Self::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }

    /// Default ridge regression algorithm
    #[must_use]
    pub fn default_ridge() -> Self {
        Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::new())
    }

    /// Default lasso regression algorithm
    #[must_use]
    pub fn default_lasso() -> Self {
        Self::Lasso(smartcore::linear::lasso::Lasso::new())
    }

    /// Default elastic net regression algorithm
    #[must_use]
    pub fn default_elastic_net() -> Self {
        Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::new())
    }

    /// Default random forest regression algorithm
    #[must_use]
    pub fn default_random_forest() -> Self {
        Self::RandomForestRegressor(
            smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
        )
    }

    /// Default extra trees regression algorithm
    #[must_use]
    pub fn default_extra_trees_regressor() -> Self {
        Self::ExtraTreesRegressor(
            smartcore::ensemble::extra_trees_regressor::ExtraTreesRegressor::new(),
        )
    }

    /// Default decision tree regression algorithm
    #[must_use]
    pub fn default_decision_tree() -> Self {
        Self::DecisionTreeRegressor(
            smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
        )
    }

    /// Default KNN regression algorithm
    #[must_use]
    pub fn default_knn_regressor() -> Self {
        Self::KNNRegressor(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }

    /// Default support vector regression algorithm
    #[must_use]
    pub fn default_support_vector_regressor() -> Self {
        Self::SupportVectorRegressor(None)
    }

    /// Default gradient boosting regression algorithm
    #[must_use]
    pub fn default_xgboost_regressor() -> Self {
        Self::XGBoostRegressor(None)
    }

    /// Get a vector of all possible algorithms
    pub fn all_algorithms(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Vec<Self> {
        <Self as Algorithm<RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>>>::all_algorithms(
            settings,
        )
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
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<Self, Failed> {
        <Self as SupervisedTrain<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
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
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<(CrossValidationResult, Self), Failed> {
        <Self as SupervisedTrain<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
        >>::cv(self, x, y, settings)
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    Algorithm<RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>>
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
            Self::DecisionTreeRegressor(model) => model.predict(x),
            Self::RandomForestRegressor(model) => model.predict(x),
            Self::ExtraTreesRegressor(model) => model.predict(x),
            Self::Linear(model) => model.predict(x),
            Self::Ridge(model) => model.predict(x),
            Self::Lasso(model) => model.predict(x),
            Self::ElasticNet(model) => model.predict(x),
            Self::KNNRegressor(model) => model.predict(x),
            Self::SupportVectorRegressor(model) => {
                let model = model
                    .as_ref()
                    .ok_or_else(|| Failed::predict("support vector regressor is not trained"))?;
                model.predict_array(x)
            }
            Self::XGBoostRegressor(model) => {
                let model = model
                    .as_ref()
                    .ok_or_else(|| Failed::predict("xgboost regressor is not trained"))?;
                convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(
                    model.predict(x)?,
                )
            }
        }
    }

    fn cross_validate_model(
        self,
        x: &Self::InputArray,
        y: &Self::OutputArray,
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
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

    fn all_algorithms(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Vec<Self> {
        let mut algorithms = vec![
            Self::default_linear(),
            Self::default_ridge(),
            Self::default_lasso(),
            Self::default_elastic_net(),
            Self::default_random_forest(),
            Self::default_decision_tree(),
        ];

        if settings.extra_trees_settings.is_some() {
            algorithms.push(Self::default_extra_trees_regressor());
        }

        if let Some(knn) = &settings.knn_regressor_settings
            && !matches!(knn.distance, Distance::Mahalanobis)
        {
            algorithms.push(Self::default_knn_regressor());
        }

        if settings.svr_settings.is_some() {
            algorithms.push(Self::default_support_vector_regressor());
        }

        if settings.xgboost_settings.is_some() {
            algorithms.push(Self::default_xgboost_regressor());
        }

        algorithms
            .retain(|algorithm| !settings.skiplist.iter().any(|skipped| skipped == algorithm));

        algorithms
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> PartialEq
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
        matches!(
            (self, other),
            (
                Self::DecisionTreeRegressor(_),
                Self::DecisionTreeRegressor(_)
            ) | (
                Self::RandomForestRegressor(_),
                Self::RandomForestRegressor(_)
            ) | (Self::Linear(_), Self::Linear(_))
                | (Self::Ridge(_), Self::Ridge(_))
                | (Self::Lasso(_), Self::Lasso(_))
                | (Self::ElasticNet(_), Self::ElasticNet(_))
                | (Self::ExtraTreesRegressor(_), Self::ExtraTreesRegressor(_))
                | (Self::KNNRegressor(_), Self::KNNRegressor(_))
                | (
                    Self::SupportVectorRegressor(_),
                    Self::SupportVectorRegressor(_)
                )
                | (Self::XGBoostRegressor(_), Self::XGBoostRegressor(_))
        )
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    OUTPUT: FloatNumber + 'static,
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
        RegressionAlgorithm::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
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
            Self::DecisionTreeRegressor(_) => write!(f, "Decision Tree Regressor"),
            Self::RandomForestRegressor(_) => write!(f, "Random Forest Regressor"),
            Self::ExtraTreesRegressor(_) => write!(f, "Extra Trees Regressor"),
            Self::Linear(_) => write!(f, "Linear Regressor"),
            Self::Ridge(_) => write!(f, "Ridge Regressor"),
            Self::Lasso(_) => write!(f, "LASSO Regressor"),
            Self::ElasticNet(_) => write!(f, "Elastic Net Regressor"),
            Self::KNNRegressor(_) => write!(f, "KNN Regressor"),
            Self::SupportVectorRegressor(_) => write!(f, "Support Vector Regressor"),
            Self::XGBoostRegressor(_) => write!(f, "XGBoost Regressor"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RegressionAlgorithm, RegressionSettings};
    use crate::DenseMatrix;
    use smartcore::error::FailedError;

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn knn_regressor_requires_settings() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[&[0.0_f64], &[1.0_f64]]).unwrap();
        let y: Vec<f64> = vec![0.0, 1.0];
        let mut settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            RegressionSettings::default();
        settings.knn_regressor_settings = None;
        let algo: RegressionAlgorithm<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            RegressionAlgorithm::default_knn_regressor();
        let err = algo
            .fit(&x, &y, &settings)
            .err()
            .expect("expected training to fail");
        assert_eq!(err.error(), FailedError::ParametersError);
    }
}
