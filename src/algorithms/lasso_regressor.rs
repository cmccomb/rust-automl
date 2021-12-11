use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix, linear::lasso::Lasso,
    model_selection::cross_validate, model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

pub(crate) struct LassoRegressorWrapper {}

impl super::ModelWrapper for LassoRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                Lasso::fit,
                x,
                y,
                settings.lasso_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::Lasso,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &Lasso::fit(x, y, settings.lasso_settings.as_ref().unwrap().clone()).unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: Lasso<f32, DenseMatrix<f32>> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
