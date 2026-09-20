//!
//! `RegressionAlgorithm` definitions and helpers

use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{
    ComparisonEntry,
    persistence::{
        as_array, as_object, finite_number, numeric_array, required_field, usize_number,
    },
    supervised::Algorithm,
};
use crate::settings::{RegressionSettings, SVRParameters, SettingsError, XGRegressorParameters};
use crate::utils::distance::{Distance, KNNRegressorDistance};
use crate::utils::kernels::SmartcoreKernel;
use serde::de::{DeserializeOwned, Error as _};
use serde::ser::Error as _;
use serde::{Deserialize, Serialize};
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
use smartcore::svm::Kernel as _;
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

// Projection of SmartCore 0.4.2's private SVR Serde representation. Any
// SmartCore upgrade must revalidate these field names and bump the artifact
// format if the persisted inference state changes.
#[derive(Deserialize)]
struct SmartcoreSVRSnapshot<INPUT> {
    instances: Option<Vec<Vec<f64>>>,
    w: Option<Vec<INPUT>>,
    b: INPUT,
}

#[derive(Serialize, Deserialize)]
struct PersistedSupportVectorRegressor<INPUT> {
    n_features: usize,
    support_vectors: Vec<Vec<f64>>,
    weights: Vec<INPUT>,
    bias: INPUT,
    kernel: smartcore::svm::Kernels,
}

impl<INPUT> PersistedSupportVectorRegressor<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn validate(&self) -> Result<(), String> {
        if self.support_vectors.len() != self.weights.len() {
            return Err("persisted SVR support-vector and weight counts differ".to_string());
        }

        let dimensions = self.n_features;
        if dimensions == 0 {
            return Err("persisted SVR support vectors have no features".to_string());
        }
        for (row, support_vector) in self.support_vectors.iter().enumerate() {
            if support_vector.len() != dimensions {
                return Err(format!(
                    "persisted SVR support vector {row} has inconsistent dimensions"
                ));
            }
            for (column, value) in support_vector.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!(
                        "persisted SVR support vector {row}, feature {column} is not finite"
                    ));
                }
            }
        }
        for (index, weight) in self.weights.iter().enumerate() {
            if !weight.to_f64().is_some_and(f64::is_finite) {
                return Err(format!("persisted SVR weight {index} is not finite"));
            }
        }
        if !self.bias.to_f64().is_some_and(f64::is_finite) {
            return Err("persisted SVR bias is not finite".to_string());
        }
        validate_svr_kernel(&self.kernel)
    }
}

fn validate_svr_kernel(kernel: &smartcore::svm::Kernels) -> Result<(), String> {
    match kernel {
        smartcore::svm::Kernels::Linear => Ok(()),
        smartcore::svm::Kernels::RBF { gamma } => {
            validate_kernel_parameter(*gamma, "RBF gamma", true)
        }
        smartcore::svm::Kernels::Polynomial {
            degree,
            gamma,
            coef0,
        } => {
            validate_kernel_parameter(*degree, "polynomial degree", true)?;
            validate_kernel_parameter(*gamma, "polynomial gamma", true)?;
            validate_kernel_parameter(*coef0, "polynomial coef0", false)
        }
        smartcore::svm::Kernels::Sigmoid { gamma, coef0 } => {
            validate_kernel_parameter(*gamma, "sigmoid gamma", false)?;
            validate_kernel_parameter(*coef0, "sigmoid coef0", false)
        }
    }
}

fn validate_kernel_parameter(
    value: Option<f64>,
    name: &str,
    must_be_positive: bool,
) -> Result<(), String> {
    let value = value.ok_or_else(|| format!("persisted SVR kernel is missing {name}"))?;
    if !value.is_finite() {
        return Err(format!("persisted SVR {name} is not finite"));
    }
    if must_be_positive && value <= 0.0 {
        return Err(format!("persisted SVR {name} must be positive"));
    }
    Ok(())
}

enum SupportVectorRegressorState<INPUT, InputArray>
where
    INPUT: RealNumber + FloatNumber + 'static,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + 'static,
{
    Trained {
        n_features: usize,
        parameters: Box<SmartcoreSVRParameters<INPUT>>,
        model: SmartcoreSVR<'static, INPUT, InputArray, Vec<INPUT>>,
    },
    Restored(PersistedSupportVectorRegressor<INPUT>),
}

/// Support vector regressor wrapper holding owned inference state.
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
    state: SupportVectorRegressorState<INPUT, InputArray>,
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
        let parameters = Box::new(params);
        let model = SmartcoreSVR::fit(x, targets, parameters.as_ref())?;
        let model = unsafe {
            mem::transmute::<
                SmartcoreSVR<'_, INPUT, InputArray, Vec<INPUT>>,
                SmartcoreSVR<'static, INPUT, InputArray, Vec<INPUT>>,
            >(model)
        };
        Ok(Self {
            state: SupportVectorRegressorState::Trained {
                n_features: x.shape().1,
                parameters,
                model,
            },
            _marker: PhantomData,
        })
    }

    fn predict_array(&self, x: &InputArray) -> Result<OutputArray, Failed> {
        if let SupportVectorRegressorState::Trained { model, .. } = &self.state {
            let predictions = model.predict(x)?;
            return convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(
                predictions,
            );
        }

        let SupportVectorRegressorState::Restored(persisted) = &self.state else {
            unreachable!();
        };
        let (rows, columns) = x.shape();
        if columns != persisted.n_features {
            return Err(Failed::predict(
                "SVR input feature count does not match the trained model",
            ));
        }
        let mut predictions = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut features = Vec::with_capacity(columns);
            for column in 0..columns {
                let value = x
                    .get((row, column))
                    .to_f64()
                    .ok_or_else(|| Failed::predict("SVR input is not representable as f64"))?;
                features.push(value);
            }

            let mut prediction = persisted.bias;
            for (support_vector, weight) in persisted.support_vectors.iter().zip(&persisted.weights)
            {
                let kernel_value = persisted.kernel.apply(&features, support_vector)?;
                let kernel_value = INPUT::from_f64(kernel_value).ok_or_else(|| {
                    Failed::predict("SVR kernel value is not representable by the input type")
                })?;
                prediction += *weight * kernel_value;
            }
            predictions.push(prediction);
        }
        convert_input_predictions_to_output_array::<INPUT, OUTPUT, OutputArray>(predictions)
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Serialize
    for OwnedSupportVectorRegressor<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + Serialize + DeserializeOwned + 'static,
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
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.state {
            SupportVectorRegressorState::Restored(persisted) => {
                persisted.validate().map_err(S::Error::custom)?;
                persisted.serialize(serializer)
            }
            SupportVectorRegressorState::Trained {
                n_features,
                parameters,
                model,
            } => {
                let kernel = parameters.kernel.clone().ok_or_else(|| {
                    S::Error::custom("trained SVR is missing its configured kernel")
                })?;
                let encoded = serde_json::to_value(model).map_err(S::Error::custom)?;
                let snapshot: SmartcoreSVRSnapshot<INPUT> =
                    serde_json::from_value(encoded).map_err(S::Error::custom)?;
                let support_vectors = snapshot
                    .instances
                    .ok_or_else(|| S::Error::custom("trained SVR is missing support vectors"))?;
                let weights = snapshot
                    .w
                    .ok_or_else(|| S::Error::custom("trained SVR is missing weights"))?;
                if support_vectors.len() != weights.len() {
                    return Err(S::Error::custom(
                        "trained SVR support-vector and weight counts differ",
                    ));
                }
                let persisted = PersistedSupportVectorRegressor {
                    n_features: *n_features,
                    support_vectors,
                    weights,
                    bias: snapshot.b,
                    kernel,
                };
                persisted.validate().map_err(S::Error::custom)?;
                persisted.serialize(serializer)
            }
        }
    }
}

impl<'de, INPUT, OUTPUT, InputArray, OutputArray> Deserialize<'de>
    for OwnedSupportVectorRegressor<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber + Deserialize<'de> + 'static,
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
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let persisted = PersistedSupportVectorRegressor::deserialize(deserializer)?;
        persisted.validate().map_err(D::Error::custom)?;
        Ok(Self {
            state: SupportVectorRegressorState::Restored(persisted),
            _marker: PhantomData,
        })
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

fn validate_linear_model(value: &serde_json::Value, name: &str) -> Result<(), String> {
    if required_field(value, "coefficients", name)?.is_null() {
        return Err(format!("{name}.coefficients is missing"));
    }
    finite_number(
        required_field(value, "intercept", name)?,
        &format!("{name}.intercept"),
    )?;
    Ok(())
}

fn validate_decision_tree(value: &serde_json::Value, name: &str) -> Result<(), String> {
    let tree = required_field(value, "tree_regressor", name)?;
    validate_base_tree(tree, &format!("{name}.tree_regressor"))
}

fn validate_forest(value: &serde_json::Value, name: &str) -> Result<(), String> {
    let forest = required_field(value, "forest_regressor", name)?;
    let trees = as_array(
        required_field(forest, "trees", &format!("{name}.forest_regressor"))?,
        &format!("{name}.forest_regressor.trees"),
    )?;
    if trees.is_empty() {
        return Err(format!("{name}.forest_regressor has no trees"));
    }
    for (index, tree) in trees.iter().enumerate() {
        validate_base_tree(tree, &format!("{name}.forest_regressor.trees[{index}]"))?;
    }
    Ok(())
}

fn validate_base_tree(value: &serde_json::Value, name: &str) -> Result<(), String> {
    as_object(
        required_field(value, "parameters", name)?,
        &format!("{name}.parameters"),
    )?;
    let nodes = as_array(
        required_field(value, "nodes", name)?,
        &format!("{name}.nodes"),
    )?;
    if nodes.is_empty() {
        return Err(format!("{name} has no root node"));
    }
    Ok(())
}

fn validate_knn(value: &serde_json::Value, name: &str) -> Result<(), String> {
    let target_state = required_field(value, "y", name)?;
    if target_state.is_null() {
        return Err(format!("{name}.y is missing"));
    }
    let target_count = target_state
        .as_array()
        .map(|_| numeric_array(target_state, &format!("{name}.y")))
        .transpose()?;

    let search_algorithms = as_object(
        required_field(value, "knn_algorithm", name)?,
        &format!("{name}.knn_algorithm"),
    )?;
    let search = search_algorithms
        .values()
        .next()
        .ok_or_else(|| format!("{name}.knn_algorithm is empty"))?;
    let data = as_array(
        required_field(search, "data", "KNN search state")?,
        "KNN search data",
    )?;
    match target_count {
        Some(targets) if data.len() != targets => {
            return Err(format!(
                "{name} has {} search rows but {targets} targets",
                data.len()
            ));
        }
        _ => {}
    }
    let mut dimensions = None;
    for (row, features) in data.iter().enumerate() {
        let row_context = format!("KNN search data row {row}");
        let width = numeric_array(features, &row_context)?;
        if width == 0 {
            return Err(format!("{row_context} has no features"));
        }
        if dimensions
            .replace(width)
            .is_some_and(|prior| prior != width)
        {
            return Err(format!("{name} search data has inconsistent row widths"));
        }
    }
    if required_field(value, "weight", name)?.is_null() {
        return Err(format!("{name}.weight is missing"));
    }
    let neighbors = usize_number(required_field(value, "k", name)?, &format!("{name}.k"))?;
    if neighbors == 0 || neighbors > data.len() {
        return Err(format!(
            "{name}.k must be between one and the number of training rows"
        ));
    }
    Ok(())
}

fn validate_svr(value: &serde_json::Value, name: &str) -> Result<(), String> {
    let support_vectors = as_array(
        required_field(value, "support_vectors", name)?,
        &format!("{name}.support_vectors"),
    )?;
    let weights = numeric_array(
        required_field(value, "weights", name)?,
        &format!("{name}.weights"),
    )?;
    let n_features = usize_number(required_field(value, "n_features", name)?, name)?;
    if n_features == 0 {
        return Err(format!("{name} has no features"));
    }
    if support_vectors.len() != weights {
        return Err(format!(
            "{name} has different support-vector and weight counts"
        ));
    }
    let mut dimensions = Some(n_features);
    for (row, support_vector) in support_vectors.iter().enumerate() {
        let row_context = format!("{name}.support_vectors[{row}]");
        let width = numeric_array(support_vector, &row_context)?;
        if width == 0 {
            return Err(format!("{row_context} has no features"));
        }
        if dimensions
            .replace(width)
            .is_some_and(|prior| prior != width)
        {
            return Err(format!(
                "{name}.support_vectors have inconsistent dimensions"
            ));
        }
    }
    finite_number(
        required_field(value, "bias", name)?,
        &format!("{name}.bias"),
    )?;
    if required_field(value, "kernel", name)?.is_null() {
        return Err(format!("{name}.kernel is missing"));
    }
    Ok(())
}

/// `RegressionAlgorithm` options
#[derive(Serialize, Deserialize)]
#[serde(
    tag = "algorithm",
    content = "model",
    rename_all = "snake_case",
    bound(
        serialize = "INPUT: Serialize + DeserializeOwned, OUTPUT: Serialize, InputArray: Serialize, OutputArray: Serialize",
        deserialize = "INPUT: Deserialize<'de>, OUTPUT: Deserialize<'de>, InputArray: Deserialize<'de>, OutputArray: Deserialize<'de>"
    )
)]
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
    #[serde(skip)]
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

    fn persistence_error(&self) -> Option<String> {
        match self {
            Self::XGBoostRegressor(_) => Some(
                "SmartCore 0.4.2 does not expose serializable XGBoost inference state".to_string(),
            ),
            Self::SupportVectorRegressor(None) => {
                Some("support vector regressor has no trained state".to_string())
            }
            _ => None,
        }
    }

    fn validate_persisted(encoded: &serde_json::Value) -> Result<(), String> {
        let algorithm = required_field(encoded, "algorithm", "regression algorithm")?
            .as_str()
            .ok_or_else(|| "regression algorithm tag must be a string".to_string())?;
        let model = required_field(encoded, "model", "regression algorithm")?;
        match algorithm {
            "linear" | "ridge" | "lasso" | "elastic_net" => validate_linear_model(model, algorithm),
            "decision_tree_regressor" => validate_decision_tree(model, algorithm),
            "random_forest_regressor" | "extra_trees_regressor" => {
                validate_forest(model, algorithm)
            }
            "k_n_n_regressor" => validate_knn(model, algorithm),
            "support_vector_regressor" => validate_svr(model, algorithm),
            "x_g_boost_regressor" => {
                Err("SmartCore 0.4.2 XGBoost state is not supported in model artifacts".to_string())
            }
            _ => Err(format!(
                "unsupported regression algorithm tag {algorithm:?}"
            )),
        }
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
