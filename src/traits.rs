//! Auto-ML for regression models

use smartcore::{
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::Matrix,
    linear::{
        elastic_net::ElasticNet, lasso::Lasso, linear_regression::LinearRegression,
        ridge_regression::RidgeRegression,
    },
    math::{distance::Distance, num::RealNumber},
    neighbors::knn_regressor::KNNRegressor,
    svm::{svr::SVR, Kernel},
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

pub(crate) trait Regressor {
    fn name(&self) -> String;
}

impl<T: RealNumber, M: Matrix<T>> Regressor for LinearRegression<T, M> {
    fn name(&self) -> String {
        "Linear Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Regressor for SVR<T, M, K> {
    fn name(&self) -> String {
        "Support Vector Regessor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> Regressor for Lasso<T, M> {
    fn name(&self) -> String {
        "LASSO Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> Regressor for RidgeRegression<T, M> {
    fn name(&self) -> String {
        "Ridge Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> Regressor for ElasticNet<T, M> {
    fn name(&self) -> String {
        "Elastic Net Regressor".to_string()
    }
}
impl<T: RealNumber> Regressor for DecisionTreeRegressor<T> {
    fn name(&self) -> String {
        "Decision Tree Regressor".to_string()
    }
}
impl<T: RealNumber, D: Distance<Vec<T>, T>> Regressor for KNNRegressor<T, D> {
    fn name(&self) -> String {
        "KNN Regressor".to_string()
    }
}
impl<T: RealNumber> Regressor for RandomForestRegressor<T> {
    fn name(&self) -> String {
        "Random Forest Regressor".to_string()
    }
}
