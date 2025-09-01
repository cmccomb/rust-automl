//! # Settings customization
//! This module contains capabilities for the detailed customization of algorithm settings.
//! ## Complete regression customization
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use automl::{
//!     algorithms::RegressionAlgorithm,
//!     settings::{
//!         DecisionTreeRegressorParameters, Distance, ElasticNetParameters, KNNAlgorithmName,
//!         KNNRegressorParameters, KNNWeightFunction, Kernel, LassoParameters,
//!         LinearRegressionParameters, LinearRegressionSolverName, Metric,
//!         RandomForestRegressorParameters, RidgeRegressionParameters, RidgeRegressionSolverName,
//!         SVRParameters,
//!     },
//! };
//!
//!  let settings = automl::RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
//!     .with_number_of_folds(3)
//!     .shuffle_data(true)
//!     .verbose(true)
//!     .skip(RegressionAlgorithm::default_random_forest())
//!     .sorted_by(Metric::RSquared)
//!     .with_linear_settings(
//!         LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
//!     )
//!     .with_lasso_settings(
//!         LassoParameters::default()
//!             .with_alpha(10.0)
//!             .with_tol(1e-10)
//!             .with_normalize(true)
//!             .with_max_iter(10_000),
//!     )
//!     .with_ridge_settings(
//!         RidgeRegressionParameters::default()
//!             .with_alpha(10.0)
//!             .with_normalize(true)
//!             .with_solver(RidgeRegressionSolverName::Cholesky),
//!     )
//!     .with_elastic_net_settings(
//!         ElasticNetParameters::default()
//!             .with_tol(1e-10)
//!             .with_normalize(true)
//!             .with_alpha(1.0)
//!             .with_max_iter(10_000)
//!             .with_l1_ratio(0.5),
//!     )
//!     .with_knn_regressor_settings(
//!         KNNRegressorParameters::default()
//!             .with_algorithm(KNNAlgorithmName::CoverTree)
//!             .with_k(3)
//!             .with_distance(Distance::Euclidean)
//!             .with_weight(KNNWeightFunction::Uniform),
//!     )
//!     .with_random_forest_regressor_settings(
//!         RandomForestRegressorParameters::default()
//!             .with_m(100)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20)
//!             .with_n_trees(100)
//!             .with_min_samples_split(20),
//!     )
//!     .with_decision_tree_regressor_settings(
//!         DecisionTreeRegressorParameters::default()
//!             .with_min_samples_split(20)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20),
//!     );
//! ```

pub use crate::utils::distance::Distance;
pub use crate::utils::kernels::Kernel;
/// Weighting functions for k-nearest neighbor (KNN) regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::neighbors::KNNWeightFunction;

/// Search algorithms for k-nearest neighbor (KNN) regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::algorithm::neighbour::KNNAlgorithmName;
/// Parameters for random forest regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters;

/// Parameters for decision tree regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters;

/// Parameters for elastic net regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::elastic_net::ElasticNetParameters;

/// Parameters for LASSO regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::lasso::LassoParameters;

/// Solvers for linear regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::linear_regression::LinearRegressionSolverName;

/// Parameters for linear regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::linear_regression::LinearRegressionParameters;

/// Parameters for ridge regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::ridge_regression::RidgeRegressionParameters;

/// Solvers for ridge regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::ridge_regression::RidgeRegressionSolverName;

/// Parameters for Gaussian naive bayes (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::naive_bayes::gaussian::GaussianNBParameters;

/// Parameters for categorical naive bayes (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::naive_bayes::categorical::CategoricalNBParameters;

/// Parameters for random forest classification (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;

/// Parameters for logistic regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::logistic_regression::LogisticRegressionParameters;

/// Parameters for logistic regression (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::linear::logistic_regression::LogisticRegressionSolverName;

/// Parameters for decision tree classification (re-export from [Smartcore](https://docs.rs/smartcore/))
pub use smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters;

mod knn_regressor_parameters;
pub use knn_regressor_parameters::KNNRegressorParameters;

mod svr_parameters;
pub use svr_parameters::SVRParameters;

mod knn_classifier_parameters;
pub use knn_classifier_parameters::KNNClassifierParameters;

mod svc_parameters;
pub use svc_parameters::SVCParameters;

mod classification_settings;
pub use classification_settings::ClassificationSettings;

mod regression_settings;
#[doc(no_inline)]
pub use regression_settings::RegressionSettings;

mod clustering_settings;
pub use clustering_settings::{ClusteringAlgorithmName, ClusteringSettings};

use std::fmt::{Display, Formatter};

/// Metrics for evaluating algorithms
#[non_exhaustive]
#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Sort by R^2
    RSquared,
    /// Sort by MAE
    MeanAbsoluteError,
    /// Sort by MSE
    MeanSquaredError,
    /// Sort by classification accuracy
    Accuracy,
    /// Sort by none
    None,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RSquared => write!(f, "R^2"),
            Self::MeanAbsoluteError => write!(f, "MAE"),
            Self::MeanSquaredError => write!(f, "MSE"),
            Self::Accuracy => write!(f, "Accuracy"),
            Self::None => panic!("A metric must be set."),
        }
    }
}

/// Options for pre-processing the data
#[derive(serde::Serialize, serde::Deserialize)]
pub enum PreProcessing {
    /// Don't do any preprocessing
    None,
    /// Add interaction terms to the data
    AddInteractions,
    /// Add polynomial terms of order n to the data
    AddPolynomial {
        /// The order of the polynomial to add (i.e., x^order)
        order: usize,
    },
    /// Replace the data with n PCA terms
    ReplaceWithPCA {
        /// The number of components to use from PCA
        number_of_components: usize,
    },
    /// Replace the data with n PCA terms
    ReplaceWithSVD {
        /// The number of components to use from PCA
        number_of_components: usize,
    },
}

impl Display for PreProcessing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::AddInteractions => write!(f, "Interaction terms added"),
            Self::AddPolynomial { order } => {
                write!(f, "Polynomial terms added (order = {order})")
            }
            Self::ReplaceWithPCA {
                number_of_components,
            } => write!(f, "Replaced with PCA features (n = {number_of_components})"),

            Self::ReplaceWithSVD {
                number_of_components,
            } => write!(f, "Replaced with SVD features (n = {number_of_components})"),
        }
    }
}

/// Final model approach
pub enum FinalAlgorithm {
    /// Do not train a final model
    None,
    /// Select the best model from the comparison set as the final model
    Best,
}
