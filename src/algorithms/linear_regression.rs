use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix, linear::linear_regression::LinearRegression,
    model_selection::cross_validate, model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

pub(crate) struct LinearRegressionWrapper {}

impl super::ModelWrapper for LinearRegressionWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                LinearRegression::fit,
                x,
                y,
                settings.linear_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::Linear,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &LinearRegression::fit(x, y, settings.linear_settings.as_ref().unwrap().clone())
                .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: LinearRegression<f32, DenseMatrix<f32>> =
            bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
