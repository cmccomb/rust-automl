use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::model_selection::cross_validate;

use crate::{Algorithm, Settings};
use smartcore::model_selection::CrossValidationResult;

pub(crate) struct LogisticRegressionWrapper {}

impl super::ModelWrapper for LogisticRegressionWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                LogisticRegression::fit,
                x,
                y,
                settings.logistic_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::LogisticRegression,
        )
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        let model: LogisticRegression<f32, DenseMatrix<f32>> =
            bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
