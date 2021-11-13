//! Auto-ML for regression models

use smartcore::{
    ensemble::{
        random_forest_classifier::RandomForestClassifier,
        random_forest_regressor::RandomForestRegressor,
    },
    linalg::Matrix,
    linear::{
        elastic_net::ElasticNet, lasso::Lasso, linear_regression::LinearRegression,
        logistic_regression::LogisticRegression, ridge_regression::RidgeRegression,
    },
    math::{distance::Distance, num::RealNumber},
    neighbors::{knn_classifier::KNNClassifier, knn_regressor::KNNRegressor},
    svm::{svc::SVC, svr::SVR, Kernel},
    tree::{
        decision_tree_classifier::DecisionTreeClassifier,
        decision_tree_regressor::DecisionTreeRegressor,
    },
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

pub(crate) trait Classifier {
    fn name(&self) -> String;
}
impl<T: RealNumber, M: Matrix<T>> Classifier for LogisticRegression<T, M> {
    fn name(&self) -> String {
        "Logistic Regression Classifier".to_string()
    }
}
impl<T: RealNumber> Classifier for RandomForestClassifier<T> {
    fn name(&self) -> String {
        "Random Forest Classifier".to_string()
    }
}
impl<T: RealNumber, D: Distance<Vec<T>, T>> Classifier for KNNClassifier<T, D> {
    fn name(&self) -> String {
        "KNN Classifier".to_string()
    }
}
impl<T: RealNumber> Classifier for DecisionTreeClassifier<T> {
    fn name(&self) -> String {
        "Decision Tree Classifier".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Classifier for SVC<T, M, K> {
    fn name(&self) -> String {
        "Suppport Vector Classifier".to_string()
    }
}
