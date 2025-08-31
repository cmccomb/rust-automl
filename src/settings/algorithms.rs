//!
//! Algorithm definitions and helpers

use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

use super::Settings;
use crate::utils::distance::Distance;
use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{Array1, Array2, MutArrayView1, MutArrayView2};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::metrics::distance::{
    euclidian::Euclidian, hamming::Hamming, manhattan::Manhattan, minkowski::Minkowski,
};
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
    /// K-nearest neighbors regressor with Euclidean distance
    KNNRegressorEuclidian(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            Euclidian<INPUT>,
        >,
    ),
    /// K-nearest neighbors regressor with Manhattan distance
    KNNRegressorManhattan(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            Manhattan<INPUT>,
        >,
    ),
    /// K-nearest neighbors regressor with Minkowski distance
    KNNRegressorMinkowski(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            Minkowski<INPUT>,
        >,
    ),
    /// K-nearest neighbors regressor with Hamming distance
    KNNRegressorHamming(
        smartcore::neighbors::knn_regressor::KNNRegressor<
            INPUT,
            OUTPUT,
            InputArray,
            OutputArray,
            Hamming<INPUT>,
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
    pub(crate) fn fit(
        self,
        x: InputArray,
        y: OutputArray,
        settings: &Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        match self {
            Self::Linear(_) => Self::Linear(
                smartcore::linear::linear_regression::LinearRegression::fit(
                    &x,
                    &y,
                    settings.linear_settings.as_ref().unwrap().clone(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::Lasso(_) => Self::Lasso(
                smartcore::linear::lasso::Lasso::fit(&x, &y, settings.lasso_settings.as_ref().unwrap().clone()).expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::Ridge(_) => Self::Ridge(
                smartcore::linear::ridge_regression::RidgeRegression::fit(
                    &x,
                    &y,
                    settings.ridge_settings.as_ref().unwrap().clone(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::ElasticNet(_) => Self::ElasticNet(
                smartcore::linear::elastic_net::ElasticNet::fit(&x, &y, settings.elastic_net_settings.as_ref().unwrap().clone()).expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::RandomForestRegressor(_) => Self::RandomForestRegressor(
                smartcore::ensemble::random_forest_regressor::RandomForestRegressor::fit(
                    &x,
                    &y,
                    settings.random_forest_regressor_settings.as_ref().unwrap().clone(),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::DecisionTreeRegressor(_) => Self::DecisionTreeRegressor(
                smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
                    &x,
                    &y,
                    settings.decision_tree_regressor_settings.as_ref().unwrap().clone(),
                )
                    .expect(
                        "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                    ),
            ),
            Self::KNNRegressorEuclidian(_) => Self::KNNRegressorEuclidian(
                smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    &x,
                    &y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default().with_k(settings.knn_regressor_settings.as_ref().unwrap().k).with_algorithm(settings.knn_regressor_settings.as_ref().unwrap().algorithm.clone()).with_weight(settings.knn_regressor_settings.as_ref().unwrap().weight.clone()).with_distance(Euclidian::new()),
                )
                    .expect(
                        "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                    ),
            ),
            Self::KNNRegressorManhattan(_) => Self::KNNRegressorManhattan(
                smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    &x,
                    &y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default().with_k(settings.knn_regressor_settings.as_ref().unwrap().k).with_algorithm(settings.knn_regressor_settings.as_ref().unwrap().algorithm.clone()).with_weight(settings.knn_regressor_settings.as_ref().unwrap().weight.clone()).with_distance(Manhattan::new()),
                )
                .expect(
                    "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
            ),
            Self::KNNRegressorMinkowski(_) => {
                let p = match settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .distance
                {
                    Distance::Minkowski(p) => p,
                    _ => unreachable!("Minkowski variant without Minkowski distance"),
                };
                Self::KNNRegressorMinkowski(
                    smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                        &x,
                        &y,
                        smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                            .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                            .with_algorithm(
                                settings
                                    .knn_regressor_settings
                                    .as_ref()
                                    .unwrap()
                                    .algorithm
                                    .clone(),
                            )
                            .with_weight(
                                settings
                                    .knn_regressor_settings
                                    .as_ref()
                                    .unwrap()
                                    .weight
                                    .clone(),
                            )
                            .with_distance(Minkowski::new(p)),
                    )
                    .expect(
                        "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                    ),
                )
            }
            Self::KNNRegressorHamming(_) => Self::KNNRegressorHamming(
                smartcore::neighbors::knn_regressor::KNNRegressor::fit(
                    &x,
                    &y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default().with_k(settings.knn_regressor_settings.as_ref().unwrap().k).with_algorithm(settings.knn_regressor_settings.as_ref().unwrap().algorithm.clone()).with_weight(settings.knn_regressor_settings.as_ref().unwrap().weight.clone()).with_distance(Hamming::new()),
                )
                    .expect(
                        "Error during training. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                    ),
            )
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
                    smartcore::linear::linear_regression::LinearRegression::<INPUT, OUTPUT, InputArray, OutputArray>::new(),
                    x,
                    y,
                    settings.linear_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_linear().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::Ridge(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::ridge_regression::RidgeRegression::<INPUT, OUTPUT, InputArray, OutputArray>::new(),
                    x,
                    y,
                    settings.ridge_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_ridge().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::Lasso(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::lasso::Lasso::<INPUT, OUTPUT, InputArray, OutputArray>::new(),
                    x,
                    y,
                    settings.lasso_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_lasso().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::ElasticNet(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::linear::elastic_net::ElasticNet::<INPUT, OUTPUT, InputArray, OutputArray>::new(),
                    x,
                    y,
                    settings.elastic_net_settings.as_ref().unwrap().clone(),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                Algorithm::default_elastic_net().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::RandomForestRegressor(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::ensemble::random_forest_regressor::RandomForestRegressor::<INPUT, OUTPUT, InputArray, OutputArray>::new(),
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
                Algorithm::default_random_forest().fit(x.clone(), y.clone(), settings),
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
                Algorithm::default_decision_tree().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::KNNRegressorEuclidian(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::neighbors::knn_regressor::KNNRegressor::<
                        INPUT,
                        OUTPUT,
                        InputArray,
                        OutputArray,
                        Euclidian<INPUT>,
                    >::new(),
                    x,
                    y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                        .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Euclidian::new()),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Algorithm::default_knn_regressor().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::KNNRegressorManhattan(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::neighbors::knn_regressor::KNNRegressor::<
                        INPUT,
                        OUTPUT,
                        InputArray,
                        OutputArray,
                        Manhattan<INPUT>,
                    >::new(),
                    x,
                    y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                        .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Manhattan::new()),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Algorithm::default_knn_regressor_manhattan().fit(x.clone(), y.clone(), settings),
            ),
            Algorithm::KNNRegressorMinkowski(_) => {
                let p = match settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .distance
                {
                    Distance::Minkowski(p) => p,
                    _ => unreachable!("Minkowski variant without Minkowski distance"),
                };
                (
                    smartcore::model_selection::cross_validate(
                        smartcore::neighbors::knn_regressor::KNNRegressor::<
                            INPUT,
                            OUTPUT,
                            InputArray,
                            OutputArray,
                            Minkowski<INPUT>,
                        >::new(),
                        x,
                        y,
                        smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                            .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                            .with_algorithm(
                                settings
                                    .knn_regressor_settings
                                    .as_ref()
                                    .unwrap()
                                    .algorithm
                                    .clone(),
                            )
                            .with_weight(
                                settings
                                    .knn_regressor_settings
                                    .as_ref()
                                    .unwrap()
                                    .weight
                                    .clone(),
                            )
                            .with_distance(Minkowski::new(p)),
                        &settings.get_kfolds(),
                        &settings.get_metric(),
                    )
                    .expect(
                        "Error during cross-validation. This is likely a bug in the AutoML library",
                    ),
                    Algorithm::default_knn_regressor_minkowski().fit(x.clone(), y.clone(), settings),
                )
            }
            Algorithm::KNNRegressorHamming(_) => (
                smartcore::model_selection::cross_validate(
                    smartcore::neighbors::knn_regressor::KNNRegressor::<
                        INPUT,
                        OUTPUT,
                        InputArray,
                        OutputArray,
                        Hamming<INPUT>,
                    >::new(),
                    x,
                    y,
                    smartcore::neighbors::knn_regressor::KNNRegressorParameters::default()
                        .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Hamming::new()),
                    &settings.get_kfolds(),
                    &settings.get_metric(),
                )
                .expect(
                    "Error during cross-validation. This is likely a bug in the AutoML library",
                ),
                Algorithm::default_knn_regressor_hamming().fit(x.clone(), y.clone(), settings),
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
        let results = self.cv(x, y, settings);
        let end = Instant::now();
        (results.0, results.1, end.duration_since(start))
    }

    /// Get a vector of all possible algorithms
    pub fn all_algorithms(
        settings: &Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Vec<Self> {
        let mut algorithms = vec![
            Self::default_linear(),
            Self::default_ridge(),
            Self::default_lasso(),
            Self::default_elastic_net(),
            Self::default_random_forest(),
            Self::default_decision_tree(),
        ];

        if let Some(knn) = &settings.knn_regressor_settings {
            match knn.distance {
                Distance::Euclidean => algorithms.push(Self::default_knn_regressor()),
                Distance::Manhattan => algorithms.push(Self::default_knn_regressor_manhattan()),
                Distance::Minkowski(_) => algorithms.push(Self::default_knn_regressor_minkowski()),
                Distance::Hamming => algorithms.push(Self::default_knn_regressor_hamming()),
                Distance::Mahalanobis => {}
            }
        }

        algorithms
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
        Self::KNNRegressorEuclidian(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }

    /// Default KNN regression algorithm using Manhattan distance
    pub fn default_knn_regressor_manhattan() -> Self {
        Self::KNNRegressorManhattan(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }

    /// Default KNN regression algorithm using Minkowski distance
    pub fn default_knn_regressor_minkowski() -> Self {
        Self::KNNRegressorMinkowski(smartcore::neighbors::knn_regressor::KNNRegressor::new())
    }

    /// Default KNN regression algorithm using Hamming distance
    pub fn default_knn_regressor_hamming() -> Self {
        Self::KNNRegressorHamming(smartcore::neighbors::knn_regressor::KNNRegressor::new())
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
        match (self, other) {
            (Self::DecisionTreeRegressor(_), Self::DecisionTreeRegressor(_))
            | (Self::RandomForestRegressor(_), Self::RandomForestRegressor(_))
            | (Self::Linear(_), Self::Linear(_))
            | (Self::Ridge(_), Self::Ridge(_))
            | (Self::Lasso(_), Self::Lasso(_))
            | (Self::ElasticNet(_), Self::ElasticNet(_)) => true,
            // treat any KNN variant as equivalent
            (a, b)
                if matches!(
                    a,
                    Self::KNNRegressorEuclidian(_)
                        | Self::KNNRegressorManhattan(_)
                        | Self::KNNRegressorMinkowski(_)
                        | Self::KNNRegressorHamming(_)
                ) && matches!(
                    b,
                    Self::KNNRegressorEuclidian(_)
                        | Self::KNNRegressorManhattan(_)
                        | Self::KNNRegressorMinkowski(_)
                        | Self::KNNRegressorHamming(_)
                ) =>
            {
                true
            }
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
            Self::KNNRegressorEuclidian(_)
            | Self::KNNRegressorManhattan(_)
            | Self::KNNRegressorMinkowski(_)
            | Self::KNNRegressorHamming(_) => write!(f, "KNN Regressor"),
        }
    }
}
