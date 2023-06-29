//! Elastic Net Regressor.

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix, linear::elastic_net::ElasticNet,
    model_selection::cross_validate, model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

/// The Elastic Net Regressor.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
/// for a more in-depth description of the algorithm.
pub(crate) struct ElasticNetRegressorWrapper {}

impl super::ModelWrapper for ElasticNetRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                ElasticNet::fit,
                x,
                y,
                settings.elastic_net_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::ElasticNet,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &ElasticNet::fit(
                x,
                y,
                settings.elastic_net_settings.as_ref().unwrap().clone(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: ElasticNet<f32, DenseMatrix<f32>> = bincode::deserialize(final_model).unwrap();
        model.predict(x).unwrap()
    }
}
