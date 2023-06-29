//! Logistic Regression

use crate::{Algorithm, Settings};
use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix, linear::logistic_regression::LogisticRegression,
    model_selection::cross_validate, model_selection::CrossValidationResult,
};

/// The Logistic Regression algorithm.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
/// for a more in-depth description of the algorithm.
pub struct LogisticRegressionWrapper {}

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

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        bincode::serialize(
            &LogisticRegression::fit(x, y, settings.logistic_settings.as_ref().unwrap().clone())
                .unwrap(),
        )
        .unwrap()
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, _settings: &Settings) -> Vec<f32> {
        let model: LogisticRegression<f32, DenseMatrix<f32>> =
            bincode::deserialize(final_model).unwrap();
        model.predict(x).unwrap()
    }
}
