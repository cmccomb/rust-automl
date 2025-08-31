//! # Settings customization
//! This module contains capabilities for the detailed customization of algorithm settings.
//! ## Complete regression customization
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use automl::settings::{
//!     Algorithm, DecisionTreeRegressorParameters, Distance, ElasticNetParameters,
//!     KNNAlgorithmName, KNNRegressorParameters, KNNWeightFunction, Kernel, LassoParameters,
//!     LinearRegressionParameters, LinearRegressionSolverName, Metric,
//!     RandomForestRegressorParameters, RidgeRegressionParameters, RidgeRegressionSolverName,
//!     SVRParameters,
//!  };
//!
//!  let settings = automl::Settings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default_regression()
//!     .with_number_of_folds(3)
//!     .shuffle_data(true)
//!     .verbose(true)
//!     .skip(Algorithm::default_random_forest())
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

use std::time::{Duration, Instant};

use smartcore::model_selection::CrossValidationResult;

use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;
use std::fmt::{Display, Formatter};

mod settings_struct;
#[doc(no_inline)]
pub use settings_struct::Settings;

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
    /// Sort by none
    None,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RSquared => write!(f, "R^2"),
            Self::MeanAbsoluteError => write!(f, "MAE"),
            Self::MeanSquaredError => write!(f, "MSE"),
            Self::None => panic!("A metric must be set."),
        }
    }
}

/// Algorithm options
pub enum Algorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    /// Decision tree regressor
    DecisionTreeRegressor(
        smartcore::tree::decision_tree_regressor::DecisionTreeRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Random forest regressor
    RandomForestRegressor(
        smartcore::ensemble::random_forest_regressor::RandomForestRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Linear regressor
    Linear(
        smartcore::linear::linear_regression::LinearRegression<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Ridge regressor
    Ridge(
        smartcore::linear::ridge_regression::RidgeRegression<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
        >,
    ),
    /// Lasso regressor
    Lasso(smartcore::linear::lasso::Lasso<INPUT, OUTPUT, InputArray, OutputArray>),
    /// Elastic net regressor
    ElasticNet(smartcore::linear::elastic_net::ElasticNet<INPUT, OUTPUT, InputArray, OutputArray>),
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Algorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    /// Fit the model
    pub(crate) fn fit(self, x: InputArray, y: OutputArray) -> Self {
        match self {
            Self::Linear(_) => Self::Linear(
                smartcore::linear::linear_regression::LinearRegression::fit(
                    &x,
                    &y,
                    Default::default(),
                )
                    .expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")
            ),
            Self::Lasso(_) => Self::Lasso(smartcore::linear::lasso::Lasso::fit(&x, &y, Default::default()).expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")),
            Self::Ridge(_) => Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::fit(&x, &y, Default::default()).expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")),
            Self::ElasticNet(_) => Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::fit(&x, &y, Default::default()).expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")),
            Self::RandomForestRegressor(_) => Self::RandomForestRegressor(smartcore::ensemble::random_forest_regressor::RandomForestRegressor::fit(&x, &y, Default::default()).expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")),
            Self::DecisionTreeRegressor(_) => Self::DecisionTreeRegressor(smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(&x, &y, Default::default()).expect("Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.")),
        }
    }

    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> (
        CrossValidationResult,
        Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) {
        match self {
            Algorithm::Linear(_) =>
                (
                    smartcore::model_selection::cross_validate(
                        smartcore::linear::linear_regression::LinearRegression::new(),
                        x,
                        y,
                        settings.linear_settings.as_ref().unwrap().clone(),
                        &settings.get_kfolds(),
                        &settings.get_metric(),
                    ).expect("Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub."),
                    Algorithm::default_linear().fit(x.clone(), y.clone())
                ),
            Algorithm::Ridge(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::ridge_regression::RidgeRegression::new(),
                    x,
                    y,
                    settings.ridge_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                ).expect("Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub."),
                Algorithm::default_ridge().fit(x.clone(), y.clone())
            ),
            Algorithm::Lasso(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::lasso::Lasso::new(),
                    x,
                    y,
                    settings.lasso_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                ).expect("Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub."),
                Algorithm::default_lasso().fit(x.clone(), y.clone())
            ),
            Algorithm::ElasticNet(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::elastic_net::ElasticNet::new(),
                    x,
                    y,
                    settings.elastic_net_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                ).expect("Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub."),
                Algorithm::default_elastic_net().fit(x.clone(), y.clone())
            ),
            Algorithm::RandomForestRegressor(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
                    x,
                    y,
                    settings
                        .random_forest_regressor_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                ).expect("Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub."),
                Algorithm::default_random_forest().fit(x.clone(), y.clone())
            ),
            Algorithm::DecisionTreeRegressor(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
                    x,
                    y,
                    settings
                        .decision_tree_regressor_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                ).expect("Error during cross-validation. This is likely a bug in the AutoML library"),
                Algorithm::default_decision_tree().fit(x.clone(), y.clone())
            ),
        }
    }

    pub(crate) fn cross_validate_model(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> (
        CrossValidationResult,
        Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
        Duration,
    ) {
        let start = Instant::now();
        let results = self.cv(&x, &y, settings);
        let end = Instant::now();
        (results.0, results.1, end.duration_since(start))
    }

    /// Get a vector of all possible algorithms
    pub fn all_algorithms() -> Vec<Self> {
        vec![
            Self::default_linear(),
            Self::default_ridge(),
            Self::default_lasso(),
            Self::default_elastic_net(),
            Self::default_random_forest(),
            Self::default_decision_tree(),
        ]
    }

    /// Default linear regression algorithm
    pub fn default_linear() -> Self {
        Self::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }

    /// Default ridge regression algorithm
    pub fn default_ridge() -> Self {
        Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::new())
    }

    /// Default lasso regression algorithm
    pub fn default_lasso() -> Self {
        Self::Lasso(smartcore::linear::lasso::Lasso::new())
    }

    /// Default elastic net regression algorithm
    pub fn default_elastic_net() -> Self {
        Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::new())
    }

    /// Default random forest regression algorithm
    pub fn default_random_forest() -> Self {
        Self::RandomForestRegressor(
            smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
        )
    }

    /// Default decision tree regression algorithm
    pub fn default_decision_tree() -> Self {
        Self::DecisionTreeRegressor(
            smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
        )
    }
}

// Implement partialeq for Algorithm
impl<INPUT, OUTPUT, InputArray, OutputArray> PartialEq
    for Algorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::DecisionTreeRegressor(_), Self::DecisionTreeRegressor(_)) => true,
            (Self::RandomForestRegressor(_), Self::RandomForestRegressor(_)) => true,
            (Self::Linear(_), Self::Linear(_)) => true,
            (Self::Ridge(_), Self::Ridge(_)) => true,
            (Self::Lasso(_), Self::Lasso(_)) => true,
            (Self::ElasticNet(_), Self::ElasticNet(_)) => true,
            _ => false,
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for Algorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn default() -> Self {
        Algorithm::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for Algorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecisionTreeRegressor(_) => write!(f, "Decision Tree Regressor"),
            Self::RandomForestRegressor(_) => write!(f, "Random Forest Regressor"),
            Self::Linear(_) => write!(f, "Linear Regressor"),
            Self::Ridge(_) => write!(f, "Ridge Regressor"),
            Self::Lasso(_) => write!(f, "LASSO Regressor"),
            Self::ElasticNet(_) => write!(f, "Elastic Net Regressor"),
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
