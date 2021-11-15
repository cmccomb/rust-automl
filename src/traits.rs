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

pub(crate) trait ValidRegressor {
    fn name(&self) -> String;
}

impl<T: RealNumber, M: Matrix<T>> ValidRegressor for LinearRegression<T, M> {
    fn name(&self) -> String {
        "Linear Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> ValidRegressor for SVR<T, M, K> {
    fn name(&self) -> String {
        "Support Vector Regessor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> ValidRegressor for Lasso<T, M> {
    fn name(&self) -> String {
        "LASSO Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> ValidRegressor for RidgeRegression<T, M> {
    fn name(&self) -> String {
        "Ridge Regressor".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>> ValidRegressor for ElasticNet<T, M> {
    fn name(&self) -> String {
        "Elastic Net Regressor".to_string()
    }
}
impl<T: RealNumber> ValidRegressor for DecisionTreeRegressor<T> {
    fn name(&self) -> String {
        "Decision Tree Regressor".to_string()
    }
}
impl<T: RealNumber, D: Distance<Vec<T>, T>> ValidRegressor for KNNRegressor<T, D> {
    fn name(&self) -> String {
        "KNN Regressor".to_string()
    }
}
impl<T: RealNumber> ValidRegressor for RandomForestRegressor<T> {
    fn name(&self) -> String {
        "Random Forest Regressor".to_string()
    }
}

pub(crate) trait ValidClassifier {
    fn name(&self) -> String;
}
impl<T: RealNumber, M: Matrix<T>> ValidClassifier for LogisticRegression<T, M> {
    fn name(&self) -> String {
        "Logistic Regression Classifier".to_string()
    }
}
impl<T: RealNumber> ValidClassifier for RandomForestClassifier<T> {
    fn name(&self) -> String {
        "Random Forest Classifier".to_string()
    }
}
impl<T: RealNumber, D: Distance<Vec<T>, T>> ValidClassifier for KNNClassifier<T, D> {
    fn name(&self) -> String {
        "KNN Classifier".to_string()
    }
}
impl<T: RealNumber> ValidClassifier for DecisionTreeClassifier<T> {
    fn name(&self) -> String {
        "Decision Tree Classifier".to_string()
    }
}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> ValidClassifier for SVC<T, M, K> {
    fn name(&self) -> String {
        "Suppport Vector Classifier".to_string()
    }
}

#[derive(PartialEq)]
pub(crate) enum Status {
    Starting,
    DataLoaded,
    ModelsCompared,
    FinalModelTrained,
}
