use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::cross_validate;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

use crate::{Algorithm, Settings};
use smartcore::model_selection::CrossValidationResult;

pub(crate) struct DecisionTreeClassifierWrapper {}

impl super::ModelWrapper for DecisionTreeClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                DecisionTreeClassifier::fit,
                x,
                y,
                settings
                    .decision_tree_classifier_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::DecisionTreeClassifier,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        todo!()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        let model: DecisionTreeClassifier<f32> = bincode::deserialize(&*final_model).unwrap();
        model.predict(x).unwrap()
    }
}
