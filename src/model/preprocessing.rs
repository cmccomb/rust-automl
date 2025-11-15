//! Utilities for data preprocessing.

use crate::model::error::ModelError;
use crate::settings::{
    CategoricalEncoderParams, CategoricalEncoding, ColumnFilterParams, ColumnSelector,
    ImputeParams, ImputeStrategy, MinMaxParams, PowerTransform, PowerTransformParams,
    PreprocessingPipeline, PreprocessingStep, RobustScaleParams, ScaleParams, ScaleStrategy,
    SettingsError, StandardizeParams,
};
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

use core::{cmp::Ordering, marker::PhantomData};

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

impl<INPUT, InputArray> Default for Preprocessor<INPUT, InputArray>
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
    fn default() -> Self {
        Self::new()
    }
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            trained_steps: Vec::new(),
        }
    }

    /// Fit preprocessing state (if required) and return a transformed copy of the
    /// training matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if any preprocessing step fails to fit or transform the
    /// provided data.
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
        for step in pipeline.steps() {
            data = self.fit_step(data, step)?;
        }
        Ok(data)
    }

    /// Apply preprocessing to inference data.
    ///
    /// # Errors
    ///
    /// Returns an error if any preprocessing step fails to transform the data
    /// using the previously fitted state.
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
        step: &PreprocessingStep,
    ) -> Result<InputArray, SettingsError> {
        match step {
            PreprocessingStep::AddInteractions => {
                self.trained_steps
                    .push(TrainedStep::Stateless(step.clone()));
                interaction_features(data).map_err(Self::feature_error_to_settings)
            }
            PreprocessingStep::AddPolynomial { order } => {
                self.trained_steps
                    .push(TrainedStep::Stateless(step.clone()));
                polynomial_features(data, *order).map_err(Self::feature_error_to_settings)
            }
            PreprocessingStep::ReplaceWithPCA {
                number_of_components,
            } => self.fit_pca_step(&data, *number_of_components),
            PreprocessingStep::ReplaceWithSVD {
                number_of_components,
            } => self.fit_svd_step(&data, *number_of_components),
            PreprocessingStep::Standardize(params) => self.fit_standardize_step(data, *params),
            PreprocessingStep::Scale(params) => self.fit_scale_step(data, params),
            PreprocessingStep::Impute(params) => self.fit_impute_step(data, params),
            PreprocessingStep::EncodeCategorical(params) => self.fit_categorical_step(data, params),
            PreprocessingStep::PowerTransform(params) => {
                self.fit_power_transform_step(data, params)
            }
            PreprocessingStep::FilterColumns(params) => self.fit_column_filter_step(data, params),
        }
    }

    fn apply_step(
        step: &TrainedStep<INPUT, InputArray>,
        data: InputArray,
    ) -> Result<InputArray, ModelError> {
        match step {
            TrainedStep::Stateless(stateless) => Self::apply_stateless(stateless, data),
            TrainedStep::Pca(pca) => pca
                .transform(&data)
                .map_err(|err| ModelError::Inference(err.to_string())),
            TrainedStep::Svd(svd) => svd
                .transform(&data)
                .map_err(|err| ModelError::Inference(err.to_string())),
            TrainedStep::Standardize(state) => state.transform_owned(data),
            TrainedStep::Scale(state) => state.transform_owned(data),
            TrainedStep::Impute(state) => state.transform_owned(data),
            TrainedStep::Categorical(state) => state.transform_owned(data),
            TrainedStep::PowerTransform(state) => state.transform_owned(data),
            TrainedStep::ColumnFilter(state) => state.transform_owned(data),
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

    fn fit_scale_step(
        &mut self,
        mut data: InputArray,
        params: &ScaleParams,
    ) -> Result<InputArray, SettingsError> {
        let state = ScaleState::fit(&mut data, params)?;
        self.trained_steps.push(TrainedStep::Scale(state));
        Ok(data)
    }

    fn fit_impute_step(
        &mut self,
        mut data: InputArray,
        params: &ImputeParams,
    ) -> Result<InputArray, SettingsError> {
        let state = ImputerState::fit(&mut data, params)?;
        self.trained_steps.push(TrainedStep::Impute(state));
        Ok(data)
    }

    fn fit_categorical_step(
        &mut self,
        data: InputArray,
        params: &CategoricalEncoderParams,
    ) -> Result<InputArray, SettingsError> {
        let (transformed, state) = CategoricalState::fit_and_transform(data, params)?;
        self.trained_steps.push(TrainedStep::Categorical(state));
        Ok(transformed)
    }

    fn fit_power_transform_step(
        &mut self,
        mut data: InputArray,
        params: &PowerTransformParams,
    ) -> Result<InputArray, SettingsError> {
        let state = PowerTransformState::fit(&mut data, params)?;
        self.trained_steps.push(TrainedStep::PowerTransform(state));
        Ok(data)
    }

    fn fit_column_filter_step(
        &mut self,
        data: InputArray,
        params: &ColumnFilterParams,
    ) -> Result<InputArray, SettingsError> {
        let (transformed, state) = ColumnFilterState::<INPUT>::fit::<InputArray>(data, params)?;
        self.trained_steps.push(TrainedStep::ColumnFilter(state));
        Ok(transformed)
    }

    fn apply_stateless(
        step: &PreprocessingStep,
        data: InputArray,
    ) -> Result<InputArray, ModelError> {
        match step {
            PreprocessingStep::AddInteractions => {
                interaction_features(data).map_err(Self::feature_error_to_model)
            }
            PreprocessingStep::AddPolynomial { order } => {
                polynomial_features(data, *order).map_err(Self::feature_error_to_model)
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
    Scale(ScaleState<INPUT>),
    Impute(ImputerState<INPUT>),
    Categorical(CategoricalState<INPUT>),
    PowerTransform(PowerTransformState<INPUT>),
    ColumnFilter(ColumnFilterState<INPUT>),
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

fn resolve_columns(
    selector: &ColumnSelector,
    total_cols: usize,
) -> Result<Vec<usize>, SettingsError> {
    match selector {
        ColumnSelector::All => Ok((0..total_cols).collect()),
        ColumnSelector::Include(indices) => {
            let mut cols = Vec::with_capacity(indices.len());
            for &idx in indices {
                if idx >= total_cols {
                    return Err(SettingsError::PreProcessingFailed(
                        "column selector index out of bounds".to_string(),
                    ));
                }
                cols.push(idx);
            }
            cols.sort_unstable();
            cols.dedup();
            Ok(cols)
        }
        ColumnSelector::Exclude(indices) => {
            let mut mask = vec![false; total_cols];
            for &idx in indices {
                if idx >= total_cols {
                    return Err(SettingsError::PreProcessingFailed(
                        "column selector index out of bounds".to_string(),
                    ));
                }
                mask[idx] = true;
            }
            Ok((0..total_cols).filter(|idx| !mask[*idx]).collect())
        }
    }
}

fn approx_zero<INPUT>(value: INPUT) -> bool
where
    INPUT: RealNumber + FloatNumber,
{
    value.abs() <= INPUT::epsilon()
}

fn collect_valid_values<INPUT, InputArray>(data: &InputArray, column: usize) -> Vec<INPUT>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (rows, cols) = data.shape();
    if column >= cols {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let value = *data.get((row, column));
        if !value.is_nan() {
            values.push(value);
        }
    }
    values
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percentile<INPUT>(values: &[INPUT], fraction: f64) -> Result<INPUT, SettingsError>
where
    INPUT: RealNumber + FloatNumber,
{
    if values.is_empty() {
        return Err(SettingsError::PreProcessingFailed(
            "cannot compute percentile of empty set".to_string(),
        ));
    }
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let clamped = fraction.clamp(0.0, 1.0);
    let position = clamped * ((values.len() - 1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return Ok(values[lower]);
    }
    let weight = position - (lower as f64);
    let lower_value = values[lower];
    let upper_value = values[upper];
    let diff = upper_value - lower_value;
    let weight_input = convert_f64_to_input::<INPUT>(weight)?;
    Ok(lower_value + diff * weight_input)
}

fn convert_f64_to_input<INPUT>(value: f64) -> Result<INPUT, SettingsError>
where
    INPUT: RealNumber + FloatNumber,
{
    INPUT::from_f64(value).ok_or_else(|| {
        SettingsError::PreProcessingFailed("cannot convert parameter to input type".to_string())
    })
}

fn convert_usize_to_input<INPUT>(value: usize) -> Result<INPUT, String>
where
    INPUT: RealNumber + FloatNumber,
{
    INPUT::from_usize(value).ok_or_else(|| "cannot convert ordinal to input type".to_string())
}

fn most_frequent<INPUT>(values: &[INPUT]) -> Option<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    let mut counts: Vec<(INPUT, usize)> = Vec::new();
    for &value in values {
        if let Some((_, count)) = counts.iter_mut().find(|(existing, _)| *existing == value) {
            *count += 1;
        } else {
            counts.push((value, 1));
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        })
        .map(|(value, _)| value)
}

fn column_min_value<INPUT, InputArray>(
    data: &InputArray,
    column: usize,
) -> Result<INPUT, SettingsError>
where
    INPUT: RealNumber + FloatNumber,
    InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
{
    let (rows, cols) = data.shape();
    if column >= cols {
        return Err(SettingsError::PreProcessingFailed(
            "column index out of bounds".to_string(),
        ));
    }
    let mut min_value = None;
    for row in 0..rows {
        let value = *data.get((row, column));
        if value.is_nan() {
            continue;
        }
        min_value = Some(match min_value {
            Some(current) => {
                if value < current {
                    value
                } else {
                    current
                }
            }
            None => value,
        });
    }
    min_value.ok_or_else(|| {
        SettingsError::PreProcessingFailed("column contains no valid values".to_string())
    })
}

#[derive(Clone, Debug)]
struct ScaleState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    entries: Vec<ScaleEntry<INPUT>>,
}

#[derive(Clone, Debug)]
struct ScaleEntry<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    column: usize,
    data: ScaleEntryData<INPUT>,
}

#[derive(Clone, Debug)]
enum ScaleEntryData<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    Standard(StandardScaleStats<INPUT>),
    MinMax {
        data_min: INPUT,
        data_range: INPUT,
        target_min: INPUT,
        target_range: INPUT,
    },
    Robust {
        center: INPUT,
        scale: INPUT,
    },
}

#[derive(Clone, Debug)]
struct StandardScaleStats<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    params: StandardizeParams,
    mean: INPUT,
    scale: INPUT,
}

impl<INPUT> ScaleState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(data: &mut InputArray, params: &ScaleParams) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let columns = resolve_columns(&params.selector, cols)?;
        let mut entries = Vec::with_capacity(columns.len());
        match &params.strategy {
            ScaleStrategy::Standard(std_params) => {
                for column in columns {
                    let stats = StandardScaleStats::compute(data, column, *std_params)?;
                    entries.push(ScaleEntry {
                        column,
                        data: ScaleEntryData::Standard(stats),
                    });
                }
            }
            ScaleStrategy::MinMax(minmax_params) => {
                for column in columns {
                    let stats = Self::min_max_stats(data, column, minmax_params)?;
                    entries.push(ScaleEntry {
                        column,
                        data: stats,
                    });
                }
            }
            ScaleStrategy::Robust(robust_params) => {
                for column in columns {
                    let stats = Self::robust_stats(data, column, robust_params)?;
                    entries.push(ScaleEntry {
                        column,
                        data: stats,
                    });
                }
            }
        }

        let state = Self { entries };
        state
            .transform_training(data)
            .map_err(SettingsError::PreProcessingFailed)?;
        Ok(state)
    }

    fn transform_training<InputArray>(&self, data: &mut InputArray) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        self.transform_internal(data)
    }

    fn transform_owned<InputArray>(&self, mut data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        self.transform_internal(&mut data)
            .map_err(ModelError::Inference)?;
        Ok(data)
    }

    fn transform_internal<InputArray>(&self, data: &mut InputArray) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        for entry in &self.entries {
            if entry.column >= cols {
                return Err("scale column index out of bounds".to_string());
            }
            for row in 0..rows {
                let mut value = *data.get((row, entry.column));
                if value.is_nan() {
                    continue;
                }
                match &entry.data {
                    ScaleEntryData::Standard(stats) => {
                        if stats.params.with_mean {
                            value -= stats.mean;
                        }
                        if stats.params.with_std {
                            let denom = if approx_zero(stats.scale) {
                                INPUT::one()
                            } else {
                                stats.scale
                            };
                            value /= denom;
                        }
                        data.set((row, entry.column), value);
                    }
                    ScaleEntryData::MinMax {
                        data_min,
                        data_range,
                        target_min,
                        target_range,
                    } => {
                        let mut centered = value - *data_min;
                        centered /= if approx_zero(*data_range) {
                            INPUT::one()
                        } else {
                            *data_range
                        };
                        centered *= *target_range;
                        centered += *target_min;
                        data.set((row, entry.column), centered);
                    }
                    ScaleEntryData::Robust { center, scale } => {
                        let mut centered = value - *center;
                        centered /= if approx_zero(*scale) {
                            INPUT::one()
                        } else {
                            *scale
                        };
                        data.set((row, entry.column), centered);
                    }
                }
            }
        }
        Ok(())
    }

    fn min_max_stats<InputArray>(
        data: &InputArray,
        column: usize,
        params: &MinMaxParams,
    ) -> Result<ScaleEntryData<INPUT>, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (mut min_val, mut max_val) = (None, None);
        let (rows, _) = data.shape();
        for row in 0..rows {
            let value = *data.get((row, column));
            if value.is_nan() {
                continue;
            }
            min_val = Some(match min_val {
                Some(current) => {
                    if value < current {
                        value
                    } else {
                        current
                    }
                }
                None => value,
            });
            max_val = Some(match max_val {
                Some(current) => {
                    if value > current {
                        value
                    } else {
                        current
                    }
                }
                None => value,
            });
        }
        let (Some(min_val), Some(max_val)) = (min_val, max_val) else {
            return Err(SettingsError::PreProcessingFailed(
                "cannot scale columns without valid values".to_string(),
            ));
        };

        let data_range = max_val - min_val;
        let (target_min_f64, target_max_f64) = params.feature_range;
        if target_max_f64 <= target_min_f64 {
            return Err(SettingsError::PreProcessingFailed(
                "invalid min-max feature range".to_string(),
            ));
        }
        let target_min = convert_f64_to_input::<INPUT>(target_min_f64)?;
        let target_range = convert_f64_to_input::<INPUT>(target_max_f64 - target_min_f64)?;

        Ok(ScaleEntryData::MinMax {
            data_min: min_val,
            data_range,
            target_min,
            target_range,
        })
    }

    fn robust_stats<InputArray>(
        data: &InputArray,
        column: usize,
        params: &RobustScaleParams,
    ) -> Result<ScaleEntryData<INPUT>, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let mut values = collect_valid_values(data, column);
        if values.is_empty() {
            return Err(SettingsError::PreProcessingFailed(
                "cannot scale columns without valid values".to_string(),
            ));
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let (low, high) = params.quantile_range;
        if !(0.0..=100.0).contains(&low) || !(0.0..=100.0).contains(&high) || high <= low {
            return Err(SettingsError::PreProcessingFailed(
                "invalid robust scaling quantiles".to_string(),
            ));
        }
        let center = percentile(&values, 0.5)?;
        let low_value = percentile(&values, low / 100.0)?;
        let high_value = percentile(&values, high / 100.0)?;
        let scale = high_value - low_value;
        Ok(ScaleEntryData::Robust { center, scale })
    }
}

impl<INPUT> StandardScaleStats<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn compute<InputArray>(
        data: &InputArray,
        column: usize,
        params: StandardizeParams,
    ) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, _) = data.shape();
        let mut sum = INPUT::zero();
        let mut count = 0usize;
        for row in 0..rows {
            let value = *data.get((row, column));
            if value.is_nan() {
                continue;
            }
            sum += value;
            count += 1;
        }
        if count == 0 {
            return Err(SettingsError::PreProcessingFailed(
                "cannot scale columns without valid values".to_string(),
            ));
        }
        let count_input = StandardScalerState::<INPUT>::convert_size(count)?;
        let mean = sum / count_input;

        let scale = if params.with_std {
            if count <= 1 {
                INPUT::one()
            } else {
                let mut sum_sq = INPUT::zero();
                for row in 0..rows {
                    let value = *data.get((row, column));
                    if value.is_nan() {
                        continue;
                    }
                    let diff = value - mean;
                    sum_sq += diff * diff;
                }
                let denom = StandardScalerState::<INPUT>::convert_size(count - 1)?;
                let variance = sum_sq / denom;
                let std = variance.sqrt();
                if approx_zero(std) { INPUT::one() } else { std }
            }
        } else {
            INPUT::one()
        };

        Ok(Self {
            params,
            mean,
            scale,
        })
    }
}

#[derive(Clone, Debug)]
struct ImputerState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    columns: Vec<usize>,
    values: Vec<INPUT>,
}

impl<INPUT> ImputerState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(data: &mut InputArray, params: &ImputeParams) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let columns = resolve_columns(&params.selector, cols)?;
        let mut values = Vec::with_capacity(columns.len());
        for &column in &columns {
            let fill_value = Self::compute_fill_value(data, column, params.strategy)?;
            values.push(fill_value);
        }
        let state = Self { columns, values };
        state
            .transform_training(data)
            .map_err(SettingsError::PreProcessingFailed)?;
        Ok(state)
    }

    fn transform_owned<InputArray>(&self, mut data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        self.transform_training(&mut data)
            .map_err(ModelError::Inference)?;
        Ok(data)
    }

    fn transform_training<InputArray>(&self, data: &mut InputArray) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        for (idx, &column) in self.columns.iter().enumerate() {
            if column >= cols {
                return Err("imputer column index out of bounds".to_string());
            }
            let replacement = self.values[idx];
            for row in 0..rows {
                let value = *data.get((row, column));
                if value.is_nan() {
                    data.set((row, column), replacement);
                }
            }
        }
        Ok(())
    }

    fn compute_fill_value<InputArray>(
        data: &InputArray,
        column: usize,
        strategy: ImputeStrategy,
    ) -> Result<INPUT, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let mut values = collect_valid_values(data, column);
        if values.is_empty() {
            return Err(SettingsError::PreProcessingFailed(
                "cannot impute column with only missing values".to_string(),
            ));
        }
        match strategy {
            ImputeStrategy::Mean => {
                let sum = values.iter().copied().fold(INPUT::zero(), |acc, v| acc + v);
                let count = StandardScalerState::<INPUT>::convert_size(values.len())?;
                Ok(sum / count)
            }
            ImputeStrategy::Median => {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                percentile(&values, 0.5)
            }
            ImputeStrategy::MostFrequent => most_frequent(&values).ok_or_else(|| {
                SettingsError::PreProcessingFailed(
                    "cannot determine most frequent value".to_string(),
                )
            }),
        }
    }
}

#[derive(Clone, Debug)]
enum CategoricalState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    Ordinal(OrdinalEncodingState<INPUT>),
    OneHot(OneHotEncodingState<INPUT>),
}

impl<INPUT> CategoricalState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit_and_transform<InputArray>(
        data: InputArray,
        params: &CategoricalEncoderParams,
    ) -> Result<(InputArray, Self), SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        match params.encoding {
            CategoricalEncoding::Ordinal => {
                let mut working = data;
                let state = OrdinalEncodingState::fit(&working, &params.selector)?;
                state
                    .transform_training(&mut working)
                    .map_err(SettingsError::PreProcessingFailed)?;
                Ok((working, Self::Ordinal(state)))
            }
            CategoricalEncoding::OneHot { drop_first } => {
                let state = OneHotEncodingState::fit(&data, &params.selector, drop_first)?;
                let transformed = state.transform_training(data)?;
                Ok((transformed, Self::OneHot(state)))
            }
        }
    }

    fn transform_owned<InputArray>(&self, data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        match self {
            Self::Ordinal(state) => state.transform_owned(data),
            Self::OneHot(state) => state.transform_owned(data),
        }
    }
}

#[derive(Clone, Debug)]
struct OrdinalEncodingState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    columns: Vec<OrdinalColumnState<INPUT>>,
}

#[derive(Clone, Debug)]
struct OrdinalColumnState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    column: usize,
    categories: Vec<INPUT>,
}

impl<INPUT> OrdinalEncodingState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(data: &InputArray, selector: &ColumnSelector) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let columns = resolve_columns(selector, cols)?;
        let mut states = Vec::with_capacity(columns.len());
        for column in columns {
            let mut categories = collect_valid_values(data, column);
            categories.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            categories.dedup();
            if categories.is_empty() {
                categories.push(INPUT::zero());
            }
            states.push(OrdinalColumnState { column, categories });
        }
        Ok(Self { columns: states })
    }

    fn transform_training<InputArray>(&self, data: &mut InputArray) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        for column_state in &self.columns {
            if column_state.column >= cols {
                return Err("ordinal encoder column index out of bounds".to_string());
            }
            for row in 0..rows {
                let value = *data.get((row, column_state.column));
                if value.is_nan() {
                    data.set((row, column_state.column), INPUT::zero());
                    continue;
                }
                let ordinal = column_state
                    .categories
                    .iter()
                    .position(|cat| *cat == value)
                    .unwrap_or(0);
                let replacement = convert_usize_to_input::<INPUT>(ordinal)?;
                data.set((row, column_state.column), replacement);
            }
        }
        Ok(())
    }

    fn transform_owned<InputArray>(&self, mut data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        self.transform_training(&mut data)
            .map_err(ModelError::Inference)?;
        Ok(data)
    }
}

#[derive(Clone, Debug)]
struct OneHotEncodingState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    columns: Vec<OneHotColumnState<INPUT>>,
    lookup: Vec<Option<usize>>,
    output_width: usize,
}

#[derive(Clone, Debug)]
struct OneHotColumnState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    column: usize,
    categories: Vec<INPUT>,
    drop_first: bool,
}

impl<INPUT> OneHotEncodingState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(
        data: &InputArray,
        selector: &ColumnSelector,
        drop_first: bool,
    ) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let selected = resolve_columns(selector, cols)?;
        let mut columns = Vec::with_capacity(selected.len());
        for column in selected {
            let mut categories = collect_valid_values(data, column);
            categories.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            categories.dedup();
            if categories.is_empty() {
                categories.push(INPUT::zero());
            }
            columns.push(OneHotColumnState {
                column,
                categories,
                drop_first,
            });
        }
        let mut lookup = vec![None; cols];
        for (idx, column_state) in columns.iter().enumerate() {
            lookup[column_state.column] = Some(idx);
        }
        let mut output_width = cols;
        for column_state in &columns {
            let generated = column_state.output_width();
            output_width = output_width - 1 + generated;
        }
        Ok(Self {
            columns,
            lookup,
            output_width,
        })
    }

    fn transform_training<InputArray>(&self, data: InputArray) -> Result<InputArray, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let transformed = self
            .transform_internal(&data)
            .map_err(SettingsError::PreProcessingFailed)?;
        drop(data);
        Ok(transformed)
    }

    fn transform_owned<InputArray>(&self, data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let transformed = self
            .transform_internal(&data)
            .map_err(ModelError::Inference)?;
        drop(data);
        Ok(transformed)
    }

    fn transform_internal<InputArray>(&self, data: &InputArray) -> Result<InputArray, String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        if cols != self.lookup.len() {
            return Err("one-hot encoder input width mismatch".to_string());
        }
        let mut values = Vec::with_capacity(rows * self.output_width);
        for row in 0..rows {
            for col in 0..cols {
                if let Some(state_idx) = self.lookup[col] {
                    self.emit_encoded(state_idx, *data.get((row, col)), &mut values);
                } else {
                    values.push(*data.get((row, col)));
                }
            }
        }
        Ok(<InputArray as Array2<INPUT>>::from_iterator(
            values.into_iter(),
            rows,
            self.output_width,
            0,
        ))
    }

    fn emit_encoded(&self, idx: usize, value: INPUT, buffer: &mut Vec<INPUT>) {
        let state = &self.columns[idx];
        for (category_idx, category) in state.categories.iter().enumerate() {
            if state.drop_first && category_idx == 0 {
                continue;
            }
            let indicator = if value == *category {
                INPUT::one()
            } else {
                INPUT::zero()
            };
            buffer.push(indicator);
        }
    }
}

impl<INPUT> OneHotColumnState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn output_width(&self) -> usize {
        if self.drop_first {
            self.categories.len().saturating_sub(1)
        } else {
            self.categories.len()
        }
    }
}

#[derive(Clone, Debug)]
struct PowerTransformState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    columns: Vec<PowerColumnState<INPUT>>,
}

#[derive(Clone, Debug)]
enum PowerColumnState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    Log {
        column: usize,
        offset: INPUT,
    },
    BoxCox {
        column: usize,
        lambda: f64,
        shift: INPUT,
    },
}

impl<INPUT> PowerTransformState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(
        data: &mut InputArray,
        params: &PowerTransformParams,
    ) -> Result<Self, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let columns = resolve_columns(&params.selector, cols)?;
        let mut entries = Vec::with_capacity(columns.len());
        match params.transform {
            PowerTransform::Log { offset } => {
                for column in columns {
                    entries.push(Self::log_entry(data, column, offset)?);
                }
            }
            PowerTransform::BoxCox { lambda } => {
                for column in columns {
                    entries.push(Self::box_cox_entry(data, column, lambda)?);
                }
            }
        }

        let state = Self { columns: entries };
        state
            .transform_training(data)
            .map_err(SettingsError::PreProcessingFailed)?;
        Ok(state)
    }

    fn transform_owned<InputArray>(&self, mut data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        self.transform_training(&mut data)
            .map_err(ModelError::Inference)?;
        Ok(data)
    }

    fn transform_training<InputArray>(&self, data: &mut InputArray) -> Result<(), String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        for entry in &self.columns {
            let column = match entry {
                PowerColumnState::Log { column, .. } | PowerColumnState::BoxCox { column, .. } => {
                    *column
                }
            };
            if column >= cols {
                return Err("power transform column index out of bounds".to_string());
            }
            for row in 0..rows {
                let value = *data.get((row, column));
                if value.is_nan() {
                    continue;
                }
                let transformed = match entry {
                    PowerColumnState::Log { offset, .. } => {
                        let adjusted = value + *offset;
                        if adjusted <= INPUT::zero() {
                            return Err(
                                "log transform requires strictly positive values".to_string()
                            );
                        }
                        adjusted.ln()
                    }
                    PowerColumnState::BoxCox { lambda, shift, .. } => {
                        let adjusted = value + *shift;
                        if adjusted <= INPUT::zero() {
                            return Err(
                                "box-cox transform requires strictly positive values".to_string()
                            );
                        }
                        if lambda.abs() <= f64::EPSILON {
                            adjusted.ln()
                        } else {
                            let lambda_input = convert_f64_to_input::<INPUT>(*lambda)
                                .map_err(|err| err.to_string())?;
                            let numerator = adjusted.powf(lambda_input) - INPUT::one();
                            numerator / lambda_input
                        }
                    }
                };
                data.set((row, column), transformed);
            }
        }
        Ok(())
    }

    fn log_entry<InputArray>(
        data: &InputArray,
        column: usize,
        offset: f64,
    ) -> Result<PowerColumnState<INPUT>, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let mut final_offset = convert_f64_to_input::<INPUT>(offset)?;
        let min_value = column_min_value(data, column)?;
        if min_value + final_offset <= INPUT::zero() {
            let adjustment = (INPUT::zero() - (min_value + final_offset)) + INPUT::epsilon();
            final_offset += adjustment;
        }
        Ok(PowerColumnState::Log {
            column,
            offset: final_offset,
        })
    }

    fn box_cox_entry<InputArray>(
        data: &InputArray,
        column: usize,
        lambda: f64,
    ) -> Result<PowerColumnState<INPUT>, SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let min_value = column_min_value(data, column)?;
        let mut shift = INPUT::zero();
        if min_value <= INPUT::zero() {
            shift = (INPUT::zero() - min_value) + INPUT::one();
        }
        Ok(PowerColumnState::BoxCox {
            column,
            lambda,
            shift,
        })
    }
}

#[derive(Clone, Debug)]
struct ColumnFilterState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    retained: Vec<usize>,
    original_width: usize,
    _marker: PhantomData<INPUT>,
}

impl<INPUT> ColumnFilterState<INPUT>
where
    INPUT: RealNumber + FloatNumber,
{
    fn fit<InputArray>(
        data: InputArray,
        params: &ColumnFilterParams,
    ) -> Result<(InputArray, Self), SettingsError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (_, cols) = data.shape();
        let selected = resolve_columns(&params.selector, cols)?;
        let retained = if params.retain_selected {
            selected
        } else {
            let mut mask = vec![false; cols];
            for &idx in &selected {
                mask[idx] = true;
            }
            (0..cols).filter(|idx| !mask[*idx]).collect::<Vec<_>>()
        };
        if retained.is_empty() {
            return Err(SettingsError::PreProcessingFailed(
                "column filter would remove all features".to_string(),
            ));
        }
        let state = Self {
            retained: retained.clone(),
            original_width: cols,
            _marker: PhantomData,
        };
        let transformed = state
            .apply_internal(&data)
            .map_err(SettingsError::PreProcessingFailed)?;
        drop(data);
        Ok((transformed, state))
    }

    fn transform_owned<InputArray>(&self, data: InputArray) -> Result<InputArray, ModelError>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let transformed = self.apply_internal(&data).map_err(ModelError::Inference)?;
        drop(data);
        Ok(transformed)
    }

    fn apply_internal<InputArray>(&self, data: &InputArray) -> Result<InputArray, String>
    where
        InputArray: Array<INPUT, (usize, usize)> + Array2<INPUT>,
    {
        let (rows, cols) = data.shape();
        if cols != self.original_width {
            return Err("column filter input width mismatch".to_string());
        }
        let new_cols = self.retained.len();
        let mut values = Vec::with_capacity(rows * new_cols);
        for row in 0..rows {
            for &col in &self.retained {
                values.push(*data.get((row, col)));
            }
        }
        Ok(<InputArray as Array2<INPUT>>::from_iterator(
            values.into_iter(),
            rows,
            new_cols,
            0,
        ))
    }
}
