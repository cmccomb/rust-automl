use smartcore::{
    ensemble::random_forest_classifier::RandomForestClassifier,
    linalg::naive::dense_matrix::DenseMatrix, model_selection::cross_validate,
    model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

pub(crate) struct RandomForestClassifierWrapper {}

impl super::ModelWrapper for RandomForestClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                RandomForestClassifier::fit,
                x,
                y,
                settings
                    .random_forest_classifier_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::RandomForestClassifier,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &RandomForestClassifier::fit(
                x,
                y,
                settings
                    .random_forest_classifier_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: RandomForestClassifier<f32> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
