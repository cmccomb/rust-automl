//! Utilities for data preprocessing.

use crate::model::error::ModelError;
use crate::settings::{PreprocessingPipeline, PreprocessingStep, SettingsError, StandardizeParams};
use crate::utils::features::{FeatureError, interaction_features, polynomial_features};
use smartcore::{
    decomposition::{
        pca::{PCA, PCAParameters},
        svd::{SVD, SVDParameters},
    },
    error::Failed,
    linalg::{
        basic::arrays::{Array, Array2},
        traits::{
            cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
            svd::SVDDecomposable,
        },
    },
    numbers::{floatnum::FloatNumber, realnum::RealNumber},
};

/// Handles optional preprocessing steps.
pub struct Preprocessor<INPUT, InputArray>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
{
    trained_steps: Vec<TrainedStep<INPUT, InputArray>>,
}

impl<INPUT, InputArray> Preprocessor<INPUT, InputArray>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
{
    /// Create a new empty preprocessor.
    pub fn new() -> Self {
        Self {
            trained_steps: Vec::new(),
        }
    }

    /// Fit preprocessing state (if required) and return a transformed copy of the
    /// training matrix.
    pub fn fit_transform(
        &mut self,
        x: InputArray,
        pipeline: &PreprocessingPipeline,
    ) -> Result<InputArray, SettingsError> {
        self.trained_steps.clear();
        if pipeline.is_empty() {
            return Ok(x);
        }

        let mut data = x;
        for &step in pipeline.steps() {
            data = self.fit_step(data, step)?;
        }
        Ok(data)
    }

    /// Apply preprocessing to inference data.
    pub fn preprocess(&self, x: InputArray) -> Result<InputArray, ModelError> {
        let mut data = x;
        for step in &self.trained_steps {
            data = Self::apply_step(step, data)?;
        }
        Ok(data)
    }

    fn fit_step(
        &mut self,
        data: InputArray,
        step: PreprocessingStep,
    ) -> Result<InputArray, SettingsError> {
        match step {
            PreprocessingStep::AddInteractions => {
                self.trained_steps.push(TrainedStep::Stateless(step));
                interaction_features(data).map_err(Self::feature_error_to_settings)
            }
            PreprocessingStep::AddPolynomial { order } => {
                self.trained_steps.push(TrainedStep::Stateless(step));
                polynomial_features(data, order).map_err(Self::feature_error_to_settings)
            }
            PreprocessingStep::ReplaceWithPCA {
                number_of_components,
            } => self.fit_pca_step(&data, number_of_components),
            PreprocessingStep::ReplaceWithSVD {
                number_of_components,
            } => self.fit_svd_step(&data, number_of_components),
            PreprocessingStep::Standardize(params) => self.fit_standardize_step(data, params),
        }
    }

    fn apply_step(
        step: &TrainedStep<INPUT, InputArray>,
        data: InputArray,
    ) -> Result<InputArray, ModelError> {
        match step {
            TrainedStep::Stateless(stateless) => Self::apply_stateless(*stateless, data),
            TrainedStep::Pca(pca) => pca
                .transform(&data)
                .map_err(|err| ModelError::Inference(err.to_string())),
            TrainedStep::Svd(svd) => svd
                .transform(&data)
                .map_err(|err| ModelError::Inference(err.to_string())),
            TrainedStep::Standardize(state) => state.transform_owned(data),
        }
    }

    fn fit_pca_step(&mut self, data: &InputArray, n: usize) -> Result<InputArray, SettingsError> {
        let pca = PCA::fit(
            data,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .map_err(|err| Self::failed_to_settings(&err))?;
        let transformed = pca
            .transform(data)
            .map_err(|err| Self::failed_to_settings(&err))?;
        self.trained_steps.push(TrainedStep::Pca(pca));
        Ok(transformed)
    }

    fn fit_svd_step(&mut self, data: &InputArray, n: usize) -> Result<InputArray, SettingsError> {
        let svd = SVD::fit(data, SVDParameters::default().with_n_components(n))
            .map_err(|err| Self::failed_to_settings(&err))?;
        let transformed = svd
            .transform(data)
            .map_err(|err| Self::failed_to_settings(&err))?;
        self.trained_steps.push(TrainedStep::Svd(svd));
        Ok(transformed)
    }

    fn fit_standardize_step(
        &mut self,
        mut data: InputArray,
        params: StandardizeParams,
    ) -> Result<InputArray, SettingsError> {
        let scaler = StandardScalerState::fit(&mut data, params)?;
        self.trained_steps.push(TrainedStep::Standardize(scaler));
        Ok(data)
    }

    fn apply_stateless(
        step: PreprocessingStep,
        data: InputArray,
    ) -> Result<InputArray, ModelError> {
        match step {
            PreprocessingStep::AddInteractions => {
                interaction_features(data).map_err(Self::feature_error_to_model)
            }
            PreprocessingStep::AddPolynomial { order } => {
                polynomial_features(data, order).map_err(Self::feature_error_to_model)
            }
            _ => Err(ModelError::Inference(
                "stateless preprocessing step requires fitting".to_string(),
            )),
        }
    }

    fn feature_error_to_settings(err: FeatureError) -> SettingsError {
        SettingsError::PreProcessingFailed(err.to_string())
    }

    fn feature_error_to_model(err: FeatureError) -> ModelError {
        ModelError::Inference(err.to_string())
    }

    fn failed_to_settings(err: &Failed) -> SettingsError {
        SettingsError::PreProcessingFailed(err.to_string())
    }
}

enum TrainedStep<INPUT, InputArray>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
{
    Stateless(PreprocessingStep),
    Pca(PCA<INPUT, InputArray>),
    Svd(SVD<INPUT, InputArray>),
    Standardize(StandardScalerState<INPUT>),
}

#[derive(Clone, Debug)]
struct StandardScalerState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    params: StandardizeParams,
    means: Vec<INPUT>,
    stds: Vec<INPUT>,
}

impl<INPUT> StandardScalerState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(
        data: &mut InputArray,
        params: StandardizeParams,
    ) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        if rows == 0 || cols == 0 {
            return Err(SettingsError::PreProcessingFailed(
                "cannot standardize empty matrix".to_string(),
            ));
        }

        let mut means = vec![INPUT::zero(); cols];
        let mut stds = vec![INPUT::one(); cols];

        if params.with_mean || params.with_std {
            let row_count = Self::convert_size(rows)?;
            for col in 0..cols {
                if params.with_mean {
                    let mut sum = INPUT::zero();
                    for row in 0..rows {
                        sum += *data.get((row, col));
                    }
                    means[col] = sum / row_count;
                }
                if params.with_std {
                    stds[col] = Self::column_std(
                        data,
                        col,
                        rows,
                        if params.with_mean {
                            means[col]
                        } else {
                            INPUT::zero()
                        },
                    )?;
                }
            }
        }

        let state = Self {
            params,
            means,
            stds,
        };
        state.transform_training(data)?;
        Ok(state)
    }

    fn column_std<InputArray>(
        data: &InputArray,
        column: usize,
        rows: usize,
        center: INPUT,
    ) -> Result<INPUT, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        if rows <= 1 {
            return Ok(INPUT::one());
        }

        let mut sum_sq = INPUT::zero();
        for row in 0..rows {
            let diff = *data.get((row, column)) - center;
            sum_sq += diff * diff;
        }
        let denom = Self::convert_size(rows - 1)?;
        let variance = sum_sq / denom;
        let std = variance.sqrt();
        if std.abs() <= INPUT::epsilon() {
            Ok(INPUT::one())
        } else {
            Ok(std)
        }
    }

    fn transform_training<InputArray>(&self, data: &mut InputArray) -> Result<(), SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        Self::transform_internal(data, self.params, &self.means, &self.stds)
            .map_err(SettingsError::PreProcessingFailed)
    }

    fn transform_owned<InputArray>(&self, mut data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        Self::transform_internal(&mut data, self.params, &self.means, &self.stds)
            .map_err(ModelError::Inference)?;
        Ok(data)
    }

    fn transform_internal<InputArray>(
        data: &mut InputArray,
        params: StandardizeParams,
        means: &[INPUT],
        stds: &[INPUT],
    ) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        if cols != means.len() || cols != stds.len() {
            return Err("scale parameters do not match feature width".to_string());
        }

        for col in 0..cols {
            let mean = if params.with_mean {
                means[col]
            } else {
                INPUT::zero()
            };
            let scale = if params.with_std {
                stds[col]
            } else {
                INPUT::one()
            };
            for row in 0..rows {
                let mut value = *data.get((row, col));
                if params.with_mean {
                    value -= mean;
                }
                if params.with_std {
                    let denom = if scale.abs() <= INPUT::epsilon() {
                        INPUT::one()
                    } else {
                        scale
                    };
                    value /= denom;
                }
                data.set((row, col), value);
            }
        }
        Ok(())
    }

    fn convert_size(value: usize) -> Result<INPUT, SettingsError> {
        INPUT::from_usize(value).ok_or_else(|| {
            SettingsError::PreProcessingFailed("cannot convert matrix dimension".to_string())
        })
    }
}
