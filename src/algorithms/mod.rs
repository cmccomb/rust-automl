mod linear_regressor;
pub(crate) use linear_regressor::LinearRegressorWrapper;
mod elastic_net_regressor;
pub(crate) use elastic_net_regressor::ElasticNetRegressorWrapper;
mod lasso_regressor;
pub(crate) use lasso_regressor::LassoRegressorWrapper;
mod knn_regressor;
pub(crate) use knn_regressor::KNNRegressorWrapper;
mod ridge_regressor;
pub(crate) use ridge_regressor::RidgeRegressorWrapper;
mod logistic_regression;
pub(crate) use logistic_regression::LogisticRegressionWrapper;
mod random_forest_classifier;
pub(crate) use random_forest_classifier::RandomForestClassifierWrapper;
mod random_forest_regressor;
pub(crate) use random_forest_regressor::RandomForestRegressorWrapper;
mod knn_classifier;
pub(crate) use knn_classifier::KNNClassifierWrapper;
mod decision_tree_classifier;
pub(crate) use decision_tree_classifier::DecisionTreeClassifierWrapper;
mod decision_tree_regressor;
pub(crate) use decision_tree_regressor::DecisionTreeRegressorWrapper;
mod gaussian_naive_bayes_classifier;
pub(crate) use gaussian_naive_bayes_classifier::GaussianNaiveBayesClassifierWrapper;
mod categorical_naive_bayes_classifier;
pub(crate) use categorical_naive_bayes_classifier::CategoricalNaiveBayesClassifierWrapper;
mod support_vector_classifier;
pub(crate) use support_vector_classifier::SupportVectorClassifierWrapper;
mod support_vector_regressor;
pub(crate) use support_vector_regressor::SupportVectorRegressorWrapper;

use crate::{Algorithm, Settings};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::CrossValidationResult;

use crate::settings::FinalModel;
use std::time::{Duration, Instant};

pub trait ModelWrapper {
    fn cv_model(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm, Duration, Vec<u8>) {
        let start = Instant::now();
        let results = Self::cv(x, y, settings);
        let end = Instant::now();
        (
            results.0,
            results.1,
            end.duration_since(start),
            match settings.final_model_approach {
                FinalModel::None => vec![],
                _ => Self::train(x, y, settings),
            },
        )
    }

    // Perform cross-validation
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm);

    // Train a model
    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8>;

    // Perform a prediction
    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32>;
}
