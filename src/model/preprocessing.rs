//! Utilities for data preprocessing.

use crate::model::error::ModelError;
use crate::settings::{PreProcessing, SettingsError};
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
    pca: Option<PCA<INPUT, InputArray>>,
    svd: Option<SVD<INPUT, InputArray>>,
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
            pca: None,
            svd: None,
        }
    }

    /// Fit preprocessing state (if required) and return a transformed copy of the
    /// training matrix.
    pub fn fit_transform(
        &mut self,
        x: InputArray,
        settings: &PreProcessing,
    ) -> Result<InputArray, SettingsError> {
        self.pca = None;
        self.svd = None;
        match settings {
            PreProcessing::None => Ok(x),
            PreProcessing::AddInteractions => {
                interaction_features(x).map_err(Self::feature_error_to_settings)
            }
            PreProcessing::AddPolynomial { order } => {
                polynomial_features(x, *order).map_err(Self::feature_error_to_settings)
            }
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => self.fit_pca(&x, *number_of_components),
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => self.fit_svd(&x, *number_of_components),
        }
    }

    /// Apply preprocessing to inference data.
    pub fn preprocess(
        &self,
        x: InputArray,
        settings: &PreProcessing,
    ) -> Result<InputArray, ModelError> {
        match settings {
            PreProcessing::None => Ok(x),
            PreProcessing::AddInteractions => {
                interaction_features(x).map_err(Self::feature_error_to_model)
            }
            PreProcessing::AddPolynomial { order } => {
                polynomial_features(x, *order).map_err(Self::feature_error_to_model)
            }
            PreProcessing::ReplaceWithPCA { .. } => self.pca_features(&x),
            PreProcessing::ReplaceWithSVD { .. } => self.svd_features(&x),
        }
    }

    fn fit_pca(&mut self, x: &InputArray, n: usize) -> Result<InputArray, SettingsError> {
        let pca = PCA::fit(
            x,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .map_err(Self::failed_to_settings)?;
        let transformed = pca.transform(x).map_err(Self::failed_to_settings)?;
        self.pca = Some(pca);
        Ok(transformed)
    }

    fn pca_features(&self, x: &InputArray) -> Result<InputArray, ModelError> {
        let pca = self
            .pca
            .as_ref()
            .ok_or_else(|| ModelError::Inference("PCA model not trained".to_string()))?;
        pca.transform(x)
            .map_err(|err| ModelError::Inference(err.to_string()))
    }

    fn fit_svd(&mut self, x: &InputArray, n: usize) -> Result<InputArray, SettingsError> {
        let svd = SVD::fit(x, SVDParameters::default().with_n_components(n))
            .map_err(Self::failed_to_settings)?;
        let transformed = svd.transform(x).map_err(Self::failed_to_settings)?;
        self.svd = Some(svd);
        Ok(transformed)
    }

    fn svd_features(&self, x: &InputArray) -> Result<InputArray, ModelError> {
        let svd = self
            .svd
            .as_ref()
            .ok_or_else(|| ModelError::Inference("SVD model not trained".to_string()))?;
        svd.transform(x)
            .map_err(|err| ModelError::Inference(err.to_string()))
    }

    fn feature_error_to_settings(err: FeatureError) -> SettingsError {
        SettingsError::PreProcessingFailed(err.to_string())
    }

    fn feature_error_to_model(err: FeatureError) -> ModelError {
        ModelError::Inference(err.to_string())
    }

    fn failed_to_settings(err: Failed) -> SettingsError {
        SettingsError::PreProcessingFailed(err.to_string())
    }
}
