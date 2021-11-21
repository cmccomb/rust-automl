//! # Settings Customization for Classification
//! This module contains capabilities for the detailed customization of algorithm settings. This
//! example shows a complete customization of the settings:
//! ```
//! use automl::classification::settings::{
//!     Algorithm, CategoricalNBParameters, DecisionTreeClassifierParameters, Distance,
//!     GaussianNBParameters, KNNAlgorithmName, KNNClassifierParameters, KNNWeightFunction, Kernel,
//!     LogisticRegressionParameters, Metric, RandomForestClassifierParameters, SVCParameters,
//! };
//!
//! let settings = automl::classification::Settings::default()
//!     .with_number_of_folds(3)
//!     .shuffle_data(true)
//!     .verbose(true)
//!     .skip(Algorithm::RandomForest)
//!     .sorted_by(Metric::Accuracy)
//!     .with_random_forest_settings(
//!         RandomForestClassifierParameters::default()
//!             .with_m(100)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20)
//!             .with_n_trees(100)
//!             .with_min_samples_split(20),
//!     )
//!     .with_logistic_settings(LogisticRegressionParameters::default())
//!     .with_svc_settings(
//!         SVCParameters::default()
//!             .with_epoch(10)
//!             .with_tol(1e-10)
//!             .with_c(1.0)
//!             .with_kernel(Kernel::Linear),
//!     )
//!     .with_decision_tree_settings(
//!         DecisionTreeClassifierParameters::default()
//!             .with_min_samples_split(20)
//!             .with_max_depth(5)
//!             .with_min_samples_leaf(20),
//!     )
//!     .with_knn_settings(
//!         KNNClassifierParameters::default()
//!             .with_algorithm(KNNAlgorithmName::CoverTree)
//!             .with_k(3)
//!             .with_distance(Distance::Euclidean)
//!             .with_weight(KNNWeightFunction::Uniform),
//!     )
//!     .with_gaussian_nb_settings(GaussianNBParameters::default().with_priors(vec![1.0, 1.0]))
//!     .with_categorical_nb_settings(CategoricalNBParameters::default().with_alpha(1.0));
//! ```
pub use crate::utils::{Distance, Kernel};

pub use smartcore::{
    algorithm::neighbour::KNNAlgorithmName,
    ensemble::random_forest_classifier::RandomForestClassifierParameters,
    linear::logistic_regression::LogisticRegressionParameters,
    naive_bayes::{categorical::CategoricalNBParameters, gaussian::GaussianNBParameters},
    neighbors::KNNWeightFunction,
    tree::decision_tree_classifier::DecisionTreeClassifierParameters,
};

use std::fmt::{Display, Formatter};

/// A struct for
pub struct SVCParameters {
    pub(crate) epoch: usize,
    pub(crate) c: f32,
    pub(crate) tol: f32,
    pub(crate) kernel: Kernel,
}

impl SVCParameters {
    /// Define the value of epsilon to use in the epsilon-SVR model.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
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

impl Default for SVCParameters {
    fn default() -> Self {
        Self {
            epoch: 2,
            c: 1.0,
            tol: 1e-3,
            kernel: Kernel::Linear,
        }
    }
}

/// Parameters for KNN Regression
pub struct KNNClassifierParameters {
    pub(crate) k: usize,
    pub(crate) weight: KNNWeightFunction,
    pub(crate) algorithm: KNNAlgorithmName,
    pub(crate) distance: Distance,
}

impl KNNClassifierParameters {
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

impl Default for KNNClassifierParameters {
    fn default() -> Self {
        Self {
            k: 3,
            weight: KNNWeightFunction::Uniform,
            algorithm: KNNAlgorithmName::CoverTree,
            distance: Distance::Euclidean,
        }
    }
}

/// An enum for sorting
#[non_exhaustive]
#[derive(PartialEq)]
pub enum Metric {
    /// Sort by accuracy
    Accuracy,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::Accuracy => write!(f, "Accuracy"),
        }
    }
}

/// An enum containing possible  classification algorithms
#[derive(PartialEq)]
pub enum Algorithm {
    /// Decision tree classifier
    DecisionTree,
    /// KNN classifier
    KNN,
    /// Random forest classifier
    RandomForest,
    /// Support vector classifier
    SVC,
    /// Logistic regression classifier
    LogisticRegression,
    /// Gaussian Naive Bayes classifier
    GaussianNaiveBayes,
    /// Categorical Naive Bayes classifier
    CategoricalNaiveBayes,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::DecisionTree => write!(f, "Decision Tree Classifier"),
            Algorithm::KNN => write!(f, "KNN Classifier"),
            Algorithm::RandomForest => write!(f, "Random Forest Classifier"),
            Algorithm::LogisticRegression => write!(f, "Logistic Regression Classifier"),
            Algorithm::SVC => write!(f, "Support Vector Classifier"),
            Algorithm::GaussianNaiveBayes => write!(f, "Gaussian Naive Bayes"),
            Algorithm::CategoricalNaiveBayes => write!(f, "Categorical Naive Bayes"),
        }
    }
}
