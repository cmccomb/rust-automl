//!
//! `RegressionAlgorithm` definitions and helpers

use std::fmt::{Display, Formatter};
use std::time::Instant;

use super::supervised_train::SupervisedTrain;
use crate::model::{ComparisonEntry, supervised::Algorithm};
use crate::settings::{RegressionSettings, WithSupervisedSettings};
use crate::settings::{RegressionSettings, WithSupervisedSettings};
use crate::utils::distance::{Distance, KNNRegressorDistance};
use smartcore::api::SupervisedEstimator;
use smartcore::error::Failed;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

/// `RegressionAlgorithm` options
pub enum RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
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
            KNNRegressorDistance<INPUT>,
        >,
    ),
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    SupervisedTrain<
        INPUT,
        OUTPUT,
        InputArray,
        OutputArray,
        RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    > for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    #[allow(clippy::too_many_lines)]
    fn fit_inner(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<Self, Failed> {
        Ok(match self {
            Self::Linear(_) => {
                Self::Linear(smartcore::linear::linear_regression::LinearRegression::fit(
                    x,
                    y,
                    settings.linear_settings.as_ref().unwrap().clone(),
                )?)
            }
            Self::Lasso(_) => Self::Lasso(smartcore::linear::lasso::Lasso::fit(
                x,
                y,
                settings.lasso_settings.as_ref().unwrap().clone(),
            )?),
            Self::Ridge(_) => {
                Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::fit(
                    x,
                    y,
                    settings.ridge_settings.as_ref().unwrap().clone(),
                )?)
            }
            Self::ElasticNet(_) => {
                Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::fit(
                    x,
                    y,
                    settings.elastic_net_settings.as_ref().unwrap().clone(),
                )?)
            }
            Self::RandomForestRegressor(_) => Self::RandomForestRegressor(
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::fit(
                    x,
                    y,
                    settings
                        .random_forest_regressor_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                )?,
            ),
            Self::DecisionTreeRegressor(_) => Self::DecisionTreeRegressor(
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
                    x,
                    y,
                    settings
                        .decision_tree_regressor_settings
                        .as_ref()
                        .unwrap()
                        .clone(),
                )?,
            ),
            Self::KNNRegressor(_) => {
                let knn_settings = settings.knn_regressor_settings.as_ref().unwrap();
                let params = knn_settings.to_regressor_params::<INPUT>();
                Self::KNNRegressor(smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    x, y, params,
                )?)
            }
        })
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::type_complexity)]
    fn cv(
        self,
        x: &InputArray,
        y: &OutputArray,
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<(CrossValidationResult, Self), Failed> {
        match self {
            RegressionAlgorithm::Linear(_) => Self::cross_validate_with(
                self,
                smartcore::linear::linear_regression::LinearRegression::new(),
                settings.linear_settings.as_ref().unwrap().clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::Ridge(_) => Self::cross_validate_with(
                self,
                smartcore::linear::ridge_regression::RidgeRegression::new(),
                settings.ridge_settings.as_ref().unwrap().clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::Lasso(_) => Self::cross_validate_with(
                self,
                smartcore::linear::lasso::Lasso::new(),
                settings.lasso_settings.as_ref().unwrap().clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::ElasticNet(_) => Self::cross_validate_with(
                self,
                smartcore::linear::elastic_net::ElasticNet::new(),
                settings.elastic_net_settings.as_ref().unwrap().clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::RandomForestRegressor(_) => Self::cross_validate_with(
                self,
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
                settings
                    .random_forest_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::DecisionTreeRegressor(_) => Self::cross_validate_with(
                self,
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
                settings
                    .decision_tree_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                x,
                y,
                settings,
                &settings.get_kfolds(),
                Self::metric(settings),
            ),
            RegressionAlgorithm::KNNRegressor(_) => {
                let knn_settings = settings.knn_regressor_settings.as_ref().unwrap();
                let params = knn_settings.to_regressor_params::<INPUT>();
                Self::cross_validate_with(
                    self,
                    smartcore::neighbors::knn_regressor::KNNRegressor::new(),
                    params,
                    x,
                    y,
                    settings,
                    &settings.get_kfolds(),
                    Self::metric(settings),
                )
            }
        }
    }

    fn metric(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> fn(&OutputArray, &OutputArray) -> f64 {
        settings.get_metric()
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    /// Default linear regression algorithm
    #[must_use]
    pub fn default_linear() -> Self {
        Self::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }

    /// Default ridge regression algorithm
    #[must_use]
    pub fn default_ridge() -> Self {
        Self::Ridge(smartcore::linear::ridge_regression::RidgeRegression::new())
    }

    /// Default lasso regression algorithm
    #[must_use]
    pub fn default_lasso() -> Self {
        Self::Lasso(smartcore::linear::lasso::Lasso::new())
    }

    /// Default elastic net regression algorithm
    #[must_use]
    pub fn default_elastic_net() -> Self {
        Self::ElasticNet(smartcore::linear::elastic_net::ElasticNet::new())
    }

    /// Default random forest regression algorithm
    #[must_use]
    pub fn default_random_forest() -> Self {
        Self::RandomForestRegressor(
            smartcore::ensemble::random_forest_regressor::RandomForestRegressor::new(),
        )
    }

    /// Default decision tree regression algorithm
    #[must_use]
    pub fn default_decision_tree() -> Self {
        Self::DecisionTreeRegressor(
            smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::new(),
        )
    }

    /// Default KNN regression algorithm
    #[must_use]
    pub fn default_knn_regressor() -> Self {
        Self::KNNRegressor(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }

    /// Get a vector of all possible algorithms
    pub fn all_algorithms(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Vec<Self> {
        <Self as Algorithm<RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>>>::all_algorithms(
            settings,
        )
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    Algorithm<RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>>
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    type Input = INPUT;
    type Output = OUTPUT;
    type InputArray = InputArray;
    type OutputArray = OutputArray;

    fn predict(&self, x: &Self::InputArray) -> Result<Self::OutputArray, Failed> {
        match self {
            Self::DecisionTreeRegressor(model) => model.predict(x),
            Self::RandomForestRegressor(model) => model.predict(x),
            Self::Linear(model) => model.predict(x),
            Self::Ridge(model) => model.predict(x),
            Self::Lasso(model) => model.predict(x),
            Self::ElasticNet(model) => model.predict(x),
            Self::KNNRegressor(model) => model.predict(x),
        }
    }

    fn cross_validate_model(
        self,
        x: &Self::InputArray,
        y: &Self::OutputArray,
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Result<ComparisonEntry<Self>, Failed> {
        let start = Instant::now();
        let results = self.cv(x, y, settings)?;
        let end = Instant::now();
        Ok(ComparisonEntry {
            result: results.0,
            algorithm: results.1,
            duration: end.duration_since(start),
        })
    }

    fn all_algorithms(
        settings: &RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Vec<Self> {
        let mut algorithms = vec![
            Self::default_linear(),
            Self::default_ridge(),
            Self::default_lasso(),
            Self::default_elastic_net(),
            Self::default_random_forest(),
            Self::default_decision_tree(),
        ];

        if let Some(knn) = &settings.knn_regressor_settings
            && !matches!(knn.distance, Distance::Mahalanobis)
        {
            algorithms.push(Self::default_knn_regressor());
        }

        algorithms
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> PartialEq
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
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
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>,
    OutputArray: MutArrayView1<OUTPUT> + Sized + Clone + Array1<OUTPUT>,
{
    fn default() -> Self {
        RegressionAlgorithm::Linear(smartcore::linear::linear_regression::LinearRegression::new())
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: MutArrayView2<INPUT>
        + Sized
        + Clone
        + Array2<INPUT>
        + QRDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
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
