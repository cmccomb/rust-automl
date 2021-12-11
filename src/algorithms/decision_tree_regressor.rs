use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    model_selection::{cross_validate, CrossValidationResult},
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

use crate::{Algorithm, Settings};

pub(crate) struct DecisionTreeRegressorWrapper {}

impl super::ModelWrapper for DecisionTreeRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                DecisionTreeRegressor::fit,
                x,
                y,
                settings
                    .decision_tree_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::DecisionTreeRegressor,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &DecisionTreeRegressor::fit(
                x,
                y,
                settings
                    .decision_tree_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: DecisionTreeRegressor<f32> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
