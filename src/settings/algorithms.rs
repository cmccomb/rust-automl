//!
//! Algorithm definitions and helpers

use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

use super::{Distance, Settings};
use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

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
    /// K-nearest neighbors regressor
    KNNRegressor(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            smartcore::metrics::distance::euclidian::Euclidian<INPUT>,
        >,
    ),
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
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::Lasso(_) => Self::Lasso(
                smartcore::linear::lasso::Lasso::fit(&x, &y, Default::default()).expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::Ridge(_) => Self::Ridge(
                smartcore::linear::ridge_regression::RidgeRegression::fit(
                    &x,
                    &y,
                    Default::default(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::ElasticNet(_) => Self::ElasticNet(
                smartcore::linear::elastic_net::ElasticNet::fit(&x, &y, Default::default()).expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::RandomForestRegressor(_) => Self::RandomForestRegressor(
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::fit(
                    &x,
                    &y,
                    Default::default(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::DecisionTreeRegressor(_) => Self::DecisionTreeRegressor(
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
                    &x,
                    &y,
                    Default::default(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::KNNRegressor(_) => Self::KNNRegressor(
                smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    &x,
                    &y,
                    Default::default(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
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
            Algorithm::Linear(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::linear_regression::LinearRegression::new(),
                    x,
                    y,
                    settings.linear_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_linear().fit(x.clone(), y.clone()),
            ),
            Algorithm::Ridge(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::ridge_regression::RidgeRegression::new(),
                    x,
                    y,
                    settings.ridge_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_ridge().fit(x.clone(), y.clone()),
            ),
            Algorithm::Lasso(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::lasso::Lasso::new(),
                    x,
                    y,
                    settings.lasso_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_lasso().fit(x.clone(), y.clone()),
            ),
            Algorithm::ElasticNet(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::elastic_net::ElasticNet::new(),
                    x,
                    y,
                    settings.elastic_net_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_elastic_net().fit(x.clone(), y.clone()),
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
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_random_forest().fit(x.clone(), y.clone()),
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
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Algorithm::default_decision_tree().fit(x.clone(), y.clone()),
            ),
            Algorithm::KNNRegressor(_) => {
                let params = settings.knn_regressor_settings.as_ref().unwrap();
                if !matches!(params.distance, Distance::Euclidean) {
                    panic!("Only Euclidean distance is supported for KNN regression");
                }
                let sc_params = smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                    .with_k(params.k)
                    .with_algorithm(params.algorithm.clone())
                    .with_weight(params.weight.clone());
                (
                    smartcore::model_selection::cross_validate(
                        smartcore::neighbors::knn_regressor::KNNRegressor::new(),
                        x,
                        y,
                        sc_params,
                        &settings.get_kfolds(),
                        &settings.get_metric(),
                    )
                    .expect(
                        "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                    ),
                    Algorithm::default_knn_regressor().fit(x.clone(), y.clone()),
                )
            }
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
        let results = self.cv(x, y, settings);
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
            Self::default_knn_regressor(),
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

    /// Default KNN regression algorithm
    pub fn default_knn_regressor() -> Self {
        Self::KNNRegressor(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }
}

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
        matches!(
            (self, other),
            (
                Self::DecisionTreeRegressor(_),
                Self::DecisionTreeRegressor(_)
            ) | (
                Self::RandomForestRegressor(_),
                Self::RandomForestRegressor(_)
            ) | (Self::Linear(_), Self::Linear(_))
                | (Self::Ridge(_), Self::Ridge(_))
                | (Self::Lasso(_), Self::Lasso(_))
                | (Self::ElasticNet(_), Self::ElasticNet(_))
                | (Self::KNNRegressor(_), Self::KNNRegressor(_))
        )
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
            Self::KNNRegressor(_) => write!(f, "KNN Regressor"),
        }
    }
}
