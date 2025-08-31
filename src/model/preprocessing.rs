//! Utilities for data preprocessing.

use crate::settings::PreProcessing;
use crate::utils::features::{interaction_features, polynomial_features};
use smartcore::{
    decomposition::{
        pca::{PCA, PCAParameters},
        svd::{SVD, SVDParameters},
    },
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

    /// Train preprocessing models based on settings.
    pub fn train(&mut self, x: &InputArray, settings: &PreProcessing) {
        match settings {
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => {
                self.train_pca(x, *number_of_components);
            }
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => {
                self.train_svd(x, *number_of_components);
            }
            _ => {}
        }
    }

    /// Apply preprocessing to data.
    pub fn preprocess(&self, x: InputArray, settings: &PreProcessing) -> InputArray {
        match settings {
            PreProcessing::None => x,
            PreProcessing::AddInteractions => interaction_features(x),
            PreProcessing::AddPolynomial { order } => polynomial_features(x, *order),
            PreProcessing::ReplaceWithPCA {
                number_of_components: _,
            } => self.pca_features(&x),
            PreProcessing::ReplaceWithSVD {
                number_of_components: _,
            } => self.svd_features(&x),
        }
    }

    fn train_pca(&mut self, x: &InputArray, n: usize) {
        let pca = PCA::fit(
            x,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .expect("Could not train PCA preprocessor");
        self.pca = Some(pca);
    }

    fn pca_features(&self, x: &InputArray) -> InputArray {
        self.pca
            .as_ref()
            .expect("PCA model not trained")
            .transform(x)
            .expect("Could not transform data using PCA")
    }

    fn train_svd(&mut self, x: &InputArray, n: usize) {
        let svd = SVD::fit(x, SVDParameters::default().with_n_components(n))
            .expect("Could not train SVD preprocessor");
        self.svd = Some(svd);
    }

    fn svd_features(&self, x: &InputArray) -> InputArray {
        self.svd
            .as_ref()
            .expect("SVD model not trained")
            .transform(x)
            .expect("Could not transform data using SVD")
    }
}
