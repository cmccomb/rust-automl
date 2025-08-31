//! Training routines for supervised models.

use super::SupervisedModel;
use crate::settings::{Algorithm, Metric, PreProcessing};
use smartcore::linalg::basic::arrays::{Array, Array1, Array2, MutArrayView1};
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::{floatnum::FloatNumber, realnum::RealNumber};
use std::time::Duration;

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
    /// Runs a model comparison and trains a final model.
    /// ```
    /// # use automl::{regression_testing_data, Settings};
    /// # use automl::supervised_model::SupervisedModel;
    /// # use smartcore::linalg::basic::matrix::DenseMatrix;
    /// # let (x, y) = regression_testing_data();
    /// let mut model = SupervisedModel::new(
    ///     x, y,
    ///     Settings::default_regression()
    /// #       .only(automl::settings::Algorithm::default_linear())
    /// );
    /// model.train();
    /// ```
    pub fn train(&mut self) {
        // Train any necessary preprocessing
        if let PreProcessing::ReplaceWithPCA {
            number_of_components,
        } = self.settings.preprocessing
        {
            self.train_pca(&self.x_train.clone(), number_of_components);
        }
        if let PreProcessing::ReplaceWithSVD {
            number_of_components,
        } = self.settings.preprocessing
        {
            self.train_svd(&self.x_train.clone(), number_of_components);
        }

        // Iterate over variants in Algorithm
        for alg in Algorithm::all_algorithms() {
            if !self.settings.skiplist.contains(&alg) {
                self.record_trained_model(alg.cross_validate_model(
                    &self.x_train,
                    &self.y_train,
                    &self.settings,
                ));
            }
        }

        // if let FinalAlgorithm::Blending {
        //     algorithm,
        //     meta_training_fraction,
        //     meta_testing_fraction,
        // } = self.settings.final_model_approach
        // {
        //     self.train_blended_model(algorithm, meta_training_fraction, meta_testing_fraction);
        // }
    }

    /// Record a model in the comparison.
    fn record_trained_model(
        &mut self,
        trained_model: (
            CrossValidationResult,
            Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
            Duration,
        ),
    ) {
        self.comparison.push(trained_model);
        self.sort();
    }

    /// Sort the models in the comparison by their mean test scores.
    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            a.0.mean_test_score()
                .partial_cmp(&b.0.mean_test_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.settings.sort_by == Metric::RSquared {
            self.comparison.reverse();
        }
    }
}
