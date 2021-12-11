use smartcore::{
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::naive::dense_matrix::DenseMatrix, model_selection::cross_validate,
    model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

pub(crate) struct RandomForestRegressorWrapper {}

impl super::ModelWrapper for RandomForestRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                RandomForestRegressor::fit,
                x,
                y,
                settings
                    .random_forest_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::RandomForestRegressor,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &RandomForestRegressor::fit(
                x,
                y,
                settings
                    .random_forest_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: RandomForestRegressor<f32> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
