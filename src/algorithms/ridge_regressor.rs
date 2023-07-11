//! Ridge regression algorithm.

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix, linear::ridge_regression::RidgeRegression,
    model_selection::cross_validate, model_selection::CrossValidationResult,
};

use crate::{Algorithm, Settings};

/// The Ridge regression algorithm.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
/// for a more in-depth description of the algorithm.
pub struct RidgeRegressorWrapper {}

impl super::ModelWrapper for RidgeRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        (
            cross_validate(
                RidgeRegression::fit,
                x,
                y,
                settings.ridge_settings.as_ref().unwrap().clone(),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Algorithm::Ridge,
        )
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &RidgeRegression::fit(x, y, settings.ridge_settings.as_ref().unwrap().clone()).unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: RidgeRegression<f32, DenseMatrix<f32>> =
            bincode::deserialize(final_model).unwrap();
        model.predict(x).unwrap()
    }
}
