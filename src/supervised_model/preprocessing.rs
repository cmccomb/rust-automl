//! Data preprocessing utilities.

use super::SupervisedModel;
use crate::settings::PreProcessing;
use smartcore::decomposition::{
    pca::{PCA, PCAParameters},
    svd::{SVD, SVDParameters},
};
use smartcore::linalg::basic::arrays::{Array, Array1, Array2, MutArrayView1};
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::numbers::{floatnum::FloatNumber, realnum::RealNumber};

impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Train PCA on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    pub(super) fn train_pca(&mut self, x: &InputArray, n: usize) {
        let pca = PCA::fit(
            x,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .unwrap();
        self.preprocessing_pca = Some(pca);
    }

    /// Get PCA features for the data using the trained PCA preprocessor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    pub(super) fn pca_features(&self, x: &InputArray, _: usize) -> InputArray {
        self.preprocessing_pca
            .as_ref()
            .unwrap()
            .transform(x)
            .expect("Could not transform data using PCA")
    }

    /// Train SVD on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    pub(super) fn train_svd(&mut self, x: &InputArray, n: usize) {
        let svd = SVD::fit(x, SVDParameters::default().with_n_components(n)).unwrap();
        self.preprocessing_svd = Some(svd);
    }

    /// Get SVD features for the data.
    pub(super) fn svd_features(&self, x: &InputArray, _: usize) -> InputArray {
        self.preprocessing_svd
            .as_ref()
            .unwrap()
            .transform(x)
            .expect("Could not transform data using SVD")
    }

    /// Pre process the data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    ///
    /// # Returns
    ///
    /// * The preprocessed data
    pub(super) fn preprocess(&self, x: InputArray) -> InputArray {
        match self.settings.preprocessing {
            PreProcessing::None => x,
            PreProcessing::AddInteractions => Self::interaction_features(x),
            PreProcessing::AddPolynomial { order } => Self::polynomial_features(x, order),
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => self.pca_features(&x, number_of_components),
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => self.svd_features(&x, number_of_components),
        }
    }
}
