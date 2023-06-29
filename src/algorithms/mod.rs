//! # Algorithms
//!
//! This module contains the wrappers for the algorithms provided by this crate.
//! The algorithms are all available through the common interface of the `ModelWrapper` trait.
//!
//! The available algorithms include:
//!
//! * Classification algorithms:
//!   - Logistic Regression
//!   - Random Forest Classifier
//!   - K-Nearest Neighbors Classifier
//!   - Decision Tree Classifier
//!   - Gaussian Naive Bayes Classifier
//!   - Categorical Naive Bayes Classifier
//!   - Support Vector Classifier
//!
//! * Regression algorithms:
//!   - Linear Regression
//!   - Elastic Net Regressor
//!   - Lasso Regressor
//!   - K-Nearest Neighbors Regressor
//!   - Ridge Regressor
//!   - Random Forest Regressor
//!   - Decision Tree Regressor
//!   - Support Vector Regressor

mod linear_regressor;
pub use linear_regressor::LinearRegressorWrapper;

mod elastic_net_regressor;
pub use elastic_net_regressor::ElasticNetRegressorWrapper;

mod lasso_regressor;
pub use lasso_regressor::LassoRegressorWrapper;

mod knn_regressor;
pub use knn_regressor::KNNRegressorWrapper;

mod ridge_regressor;
pub use ridge_regressor::RidgeRegressorWrapper;

mod logistic_regression;
pub use logistic_regression::LogisticRegressionWrapper;

mod random_forest_classifier;
pub use random_forest_classifier::RandomForestClassifierWrapper;

mod random_forest_regressor;
pub use random_forest_regressor::RandomForestRegressorWrapper;

mod knn_classifier;
pub use knn_classifier::KNNClassifierWrapper;

mod decision_tree_classifier;
pub use decision_tree_classifier::DecisionTreeClassifierWrapper;

mod decision_tree_regressor;
pub use decision_tree_regressor::DecisionTreeRegressorWrapper;

mod gaussian_naive_bayes_classifier;
pub use gaussian_naive_bayes_classifier::GaussianNaiveBayesClassifierWrapper;

mod categorical_naive_bayes_classifier;
pub use categorical_naive_bayes_classifier::CategoricalNaiveBayesClassifierWrapper;

mod support_vector_classifier;
pub use support_vector_classifier::SupportVectorClassifierWrapper;

mod support_vector_regressor;
pub use support_vector_regressor::SupportVectorRegressorWrapper;

use crate::{Algorithm, Settings};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::CrossValidationResult;

use crate::settings::FinalModel;
use std::time::{Duration, Instant};

/// Trait for wrapping models
pub trait ModelWrapper {
    /// Perform cross-validation and return the results
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `y` - The output data
    /// * `settings` - The settings for the model
    ///
    /// # Returns
    ///
    /// * `CrossValidationResult<f32>` - The cross-validation results
    /// * `Algorithm` - The algorithm used
    /// * `Duration` - The time taken to perform the cross-validation
    /// * `Vec<u8>` - The final model
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

    /// Perform cross-validation
    #[allow(clippy::ptr_arg)]
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm);

    /// Train a model
    #[allow(clippy::ptr_arg)]
    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8>;

    /// Perform a prediction
    #[allow(clippy::ptr_arg)]
    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32>;
}
