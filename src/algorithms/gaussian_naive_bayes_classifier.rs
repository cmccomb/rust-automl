use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::cross_validate;
use smartcore::naive_bayes::gaussian::GaussianNB;

use crate::{Algorithm, Settings};
use smartcore::model_selection::CrossValidationResult;

pub(crate) struct GaussianNaiveBayesClassifierWrapper {}

impl super::ModelWrapper for GaussianNaiveBayesClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                GaussianNB::fit,
                x,
                y,
                settings.gaussian_nb_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::GaussianNaiveBayes,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &GaussianNB::fit(
                x,
                y,
                settings.gaussian_nb_settings.as_ref().unwrap().clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        let model: GaussianNB<f32, DenseMatrix<f32>> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
