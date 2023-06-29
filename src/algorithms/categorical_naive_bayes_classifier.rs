//! Categorical Naive Bayes Classifier.

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    model_selection::{cross_validate, CrossValidationResult},
    naive_bayes::categorical::CategoricalNB,
};

use crate::{Algorithm, Settings};

/// The Categorical Naive Bayes Classifier.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes)
/// for a more in-depth description of the algorithm.
pub(crate) struct CategoricalNaiveBayesClassifierWrapper {}

impl super::ModelWrapper for CategoricalNaiveBayesClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                CategoricalNB::fit,
                x,
                y,
                settings.categorical_nb_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::CategoricalNaiveBayes,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &CategoricalNB::fit(
                x,
                y,
                settings.categorical_nb_settings.as_ref().unwrap().clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: CategoricalNB<f32, DenseMatrix<f32>> =
            bincode::deserialize(final_model).unwrap();
        model.predict(x).unwrap()
    }
}
