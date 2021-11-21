//! # Settings Customization for Regression
//! This module contains capabilities for the detailed customization of algorithm settings. This
//! example shows a complete customization of the settings:
//! ```
//! use automl::regression::settings::{
//!     Algorithm, DecisionTreeRegressorParameters, Distance, ElasticNetParameters,
//!     KNNAlgorithmName, KNNRegressorParameters, KNNWeightFunction, Kernel, LassoParameters,
//!     LinearRegressionParameters, LinearRegressionSolverName, Metric,
//!     RandomForestRegressorParameters, RidgeRegressionParameters, RidgeRegressionSolverName,
//!     SVRParameters,
//!  };
//!
//!  let settings = automl::regression::Settings::default()
//!     .with_number_of_folds(3)
//!     .shuffle_data(true)
//!     .verbose(true)
//!     .skip(Algorithm::RandomForest)
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
//!     .with_knn_settings(
//!         KNNRegressorParameters::default()
//!             .with_algorithm(KNNAlgorithmName::CoverTree)
//!             .with_k(3)
//!             .with_distance(Distance::Euclidean)
//!             .with_weight(KNNWeightFunction::Uniform),
//!     )
//!     .with_svr_settings(
//!         SVRParameters::default()
//!             .with_eps(1e-10)
//!             .with_tol(1e-10)
//!             .with_c(1.0)
//!             .with_kernel(Kernel::Linear),
//!     )
//!     .with_random_forest_settings(
//!         RandomForestRegressorParameters::default()
//!             .with_m(100)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20)
//!             .with_n_trees(100)
//!             .with_min_samples_split(20),
//!     )
//!     .with_decision_tree_settings(
//!         DecisionTreeRegressorParameters::default()
//!             .with_min_samples_split(20)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20),
//!     );
//! ```
pub use crate::utils::{Distance, Kernel};

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

use std::fmt::{Display, Formatter};

/// Parameters for k-nearest neighbor (KNN) regression
pub struct KNNRegressorParameters {
    pub(crate) k: usize,
    pub(crate) weight: KNNWeightFunction,
    pub(crate) algorithm: KNNAlgorithmName,
    pub(crate) distance: Distance,
}

impl KNNRegressorParameters {
    /// Define the number of nearest neighbors to use
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Define the weighting function to use with KNN regresssion
    pub fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }

    /// Define the search algorithm to use with KNN regresssion
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Define the distance metric to use with KNN regresssion
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }
}

impl Default for KNNRegressorParameters {
    fn default() -> Self {
        Self {
            k: 3,
            weight: KNNWeightFunction::Uniform,
            algorithm: KNNAlgorithmName::CoverTree,
            distance: Distance::Euclidean,
        }
    }
}

/// Parameters for support vector regression
pub struct SVRParameters {
    pub(crate) eps: f32,
    pub(crate) c: f32,
    pub(crate) tol: f32,
    pub(crate) kernel: Kernel,
}

impl SVRParameters {
    /// Define the value of epsilon to use in the epsilon-SVR model.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Define the regulation penalty to use with the SVR Model
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Define the convergence tolereance to use with the SVR model
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Define which kernel to use with the SVR model
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
}

impl Default for SVRParameters {
    fn default() -> Self {
        Self {
            eps: 0.1,
            c: 1.0,
            tol: 1e-3,
            kernel: Kernel::Linear,
        }
    }
}

/// Metrics for evaluating algorithms
#[non_exhaustive]
#[derive(PartialEq)]
pub enum Metric {
    /// Sort by R^2
    RSquared,
    /// Sort by MAE
    MeanAbsoluteError,
    /// Sort by MSE
    MeanSquaredError,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::RSquared => write!(f, "R^2"),
            Metric::MeanAbsoluteError => write!(f, "MAE"),
            Metric::MeanSquaredError => write!(f, "MSE"),
        }
    }
}

/// Regression algorithm options
#[derive(PartialEq)]
pub enum Algorithm {
    /// Decision tree regressor
    DecisionTree,
    /// KNN Regressor
    KNN,
    /// Random forest regressor
    RandomForest,
    /// Linear regressor
    Linear,
    /// Ridge regressor
    Ridge,
    /// Lasso regressor
    Lasso,
    /// Elastic net regressor
    ElasticNet,
    /// Support vector regressor
    SVR,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::DecisionTree => write!(f, "Decision Tree Regressor"),
            Algorithm::KNN => write!(f, "KNN Regressor"),
            Algorithm::RandomForest => write!(f, "Random Forest Regressor"),
            Algorithm::Linear => write!(f, "Linear Regressor"),
            Algorithm::Ridge => write!(f, "Ridge Regressor"),
            Algorithm::Lasso => write!(f, "LASSO Regressor"),
            Algorithm::ElasticNet => write!(f, "Elastic Net Regressor"),
            Algorithm::SVR => write!(f, "Support Vector Regressor"),
        }
    }
}
