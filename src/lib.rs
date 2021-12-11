#![warn(clippy::all)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(clippy::missing_docs_in_private_items)]
#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod settings;
use settings::{
    Algorithm, CategoricalNBParameters, DecisionTreeClassifierParameters,
    DecisionTreeRegressorParameters, Distance, ElasticNetParameters, GaussianNBParameters,
    KNNClassifierParameters, KNNRegressorParameters, Kernel, LassoParameters,
    LinearRegressionParameters, LinearRegressionSolverName, LogisticRegressionParameters, Metric,
    PreProcessing, RandomForestClassifierParameters, RandomForestRegressorParameters,
    RidgeRegressionParameters, RidgeRegressionSolverName, SVCParameters, SVRParameters,
};

mod algorithms;
use algorithms::{
    CategoricalNaiveBayesClassifierWrapper, DecisionTreeClassifierWrapper,
    GaussianNaiveBayesClassifierWrapper, KNNClassifierWrapper, LinearRegressionWrapper,
    LogisticRegressionWrapper, ModelWrapper, RandomForestClassifierWrapper,
    SupportVectorClassifierWrapper,
};

mod utils;
use crate::utils::{
    debug_option, elementwise_multiply, print_knn_search_algorithm, print_knn_weight_function,
    print_option,
};
use itertools::Itertools;
use smartcore::{
    dataset::Dataset,
    decomposition::{
        pca::{PCAParameters, PCA},
        svd::{SVDParameters, SVD},
    },
    ensemble::{
        random_forest_classifier::RandomForestClassifier,
        random_forest_regressor::RandomForestRegressor,
    },
    linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix},
    linear::{
        elastic_net::ElasticNet, lasso::Lasso, linear_regression::LinearRegression,
        logistic_regression::LogisticRegression, ridge_regression::RidgeRegression,
    },
    math::distance::{
        euclidian::Euclidian, hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan,
        minkowski::Minkowski, Distances,
    },
    metrics::{accuracy, mean_absolute_error, mean_squared_error, r2},
    model_selection::{cross_validate, CrossValidationResult, KFold},
    naive_bayes::{categorical::CategoricalNB, gaussian::GaussianNB},
    neighbors::{
        knn_classifier::{
            KNNClassifier, KNNClassifierParameters as SmartcoreKNNClassifierParameters,
        },
        knn_regressor::{KNNRegressor, KNNRegressorParameters as SmartcoreKNNRegressorParameters},
    },
    svm::{
        svc::{SVCParameters as SmartcoreSVCParameters, SVC},
        svr::{SVRParameters as SmartcoreSVRParameters, SVR},
        Kernels, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel,
    },
    tree::{
        decision_tree_classifier::{DecisionTreeClassifier, SplitCriterion},
        decision_tree_regressor::DecisionTreeRegressor,
    },
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    time::{Duration, Instant},
};

#[cfg(any(feature = "nd"))]
use ndarray::{Array1, Array2};

#[cfg(any(feature = "gui"))]
use eframe::{egui, epi};

#[cfg(any(feature = "csv"))]
use polars::prelude::{CsvReader, DataFrame, Float32Type, SerReader};

#[cfg(any(feature = "display"))]
use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};

#[cfg(any(feature = "display"))]
use humantime::format_duration;

/// Trains and compares supervised models
pub struct SupervisedModel {
    settings: Settings,
    x: DenseMatrix<f32>,
    y: Vec<f32>,
    number_of_classes: usize,
    comparison: Vec<Model>,
    final_model: Vec<u8>,
    current_x: Vec<f32>,
    preprocessing: (
        Option<PCA<f32, DenseMatrix<f32>>>,
        Option<SVD<f32, DenseMatrix<f32>>>,
    ),
}

impl SupervisedModel {
    /// Create a new supervised model from a csv
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = SupervisedModel::new_from_csv(
    ///     "data/diabetes.csv",
    ///     10,
    ///     true,
    ///     Settings::default_regression()
    /// );
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "csv")))]
    #[cfg(any(feature = "csv"))]
    pub fn new_from_csv(
        filepath: &str,
        target_index: usize,
        header: bool,
        settings: Settings,
    ) -> Self {
        let df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

        // Get target variables
        let target_column_name = df.get_column_names()[target_index];
        let series = df.column(target_column_name).unwrap().clone();
        let target_df = DataFrame::new(vec![series]).unwrap();
        let ndarray = target_df.to_ndarray::<Float32Type>().unwrap();
        let y = ndarray.into_raw_vec();

        // Get the rest of the data
        let features = df.drop(target_column_name).unwrap();
        let (height, width) = features.shape();
        let ndarray = features.to_ndarray::<Float32Type>().unwrap();
        let x = DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap());

        let current_x = vec![0.0; x.clone().shape().1];

        Self {
            settings,
            x,
            y: y.clone(),
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            final_model: vec![],
            current_x,
            preprocessing: (None, None),
        }
    }

    /// Create a new supervised model from a [smartcore toy dataset](https://docs.rs/smartcore/0.2.0/smartcore/dataset/index.html)
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// ```
    pub fn new_from_dataset(dataset: Dataset<f32, f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        let y = dataset.target;
        let current_x = vec![0.0; x.clone().shape().1];

        Self {
            settings,
            x,
            y: y.clone(),
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            final_model: vec![],
            current_x,
            preprocessing: (None, None),
        }
    }

    /// Create a new supervised model using vec data
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = automl::SupervisedModel::new_from_vec(
    ///     vec![vec![1.0; 5]; 5],
    ///     vec![1.0; 5],
    ///     automl::Settings::default_regression(),
    /// );    
    /// ```
    pub fn new_from_vec(x: Vec<Vec<f32>>, y: Vec<f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_2d_vec(&x);
        let current_x = vec![0.0; x.clone().shape().1];

        Self {
            settings,
            x,
            y: y.clone(),
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            final_model: vec![],
            current_x,
            preprocessing: (None, None),
        }
    }

    /// Create a new supervised model using ndarray data
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// use ndarray::{arr1, arr2};
    /// let model = automl::SupervisedModel::new_from_ndarray(
    ///     arr2(&[[1.0, 2.0], [3.0, 4.0]]),
    ///     arr1(&[1.0, 2.0]),
    ///     automl::Settings::default_regression(),
    /// );
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "nd")))]
    #[cfg(any(feature = "nd"))]
    pub fn new_from_ndarray(x: Array2<f32>, y: Array1<f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap());
        let y = y.to_vec();

        let current_x = vec![0.0; x.clone().shape().1];

        Self {
            settings,
            x,
            y: y.clone(),
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            final_model: vec![],
            current_x,
            preprocessing: (None, None),
        }
    }

    /// Runs a model comparison and trains a final model.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.auto();
    /// ```
    pub fn auto(&mut self) {
        self.compare_models();
        self.train_final_model();
    }

    /// This function compares all of the  models available in the package.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.compare_models();
    /// ```
    pub fn compare_models(&mut self) {
        // Find the right metric to use
        let metric = match self.settings.sort_by {
            Metric::RSquared => r2,
            Metric::MeanAbsoluteError => mean_absolute_error,
            Metric::MeanSquaredError => mean_squared_error,
            Metric::Accuracy => accuracy,
            Metric::None => panic!("A metric must be set."),
        };

        // Preorcess teh dadta
        self.x = self.preprocess(self.x.clone());

        // Run logistic regression
        if !self
            .settings
            .skiplist
            .contains(&Algorithm::LogisticRegression)
        {
            self.record_model(LogisticRegressionWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        // Run random forest classification
        if !self
            .settings
            .skiplist
            .contains(&Algorithm::RandomForestClassifier)
        {
            self.record_model(RandomForestClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        // Run k-nearest neighbor classifier
        if !self.settings.skiplist.contains(&Algorithm::KNNClassifier) {
            self.record_model(KNNClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::DecisionTreeClassifier)
        {
            self.record_model(DecisionTreeClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::GaussianNaiveBayes)
        {
            self.record_model(GaussianNaiveBayesClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::CategoricalNaiveBayes)
        {
            self.record_model(CategoricalNaiveBayesClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if self.number_of_classes == 2 && !self.settings.skiplist.contains(&Algorithm::SVC) {
            self.record_model(SupportVectorClassifierWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::Linear) {
            self.record_model(LinearRegressionWrapper::cv_model(
                &self.x,
                &self.y,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::SVR) {
            let start = Instant::now();
            let cv = match self.settings.svr_settings.as_ref().unwrap().kernel {
                Kernel::Linear => cross_validate(
                    SVR::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_kernel(Kernels::linear()),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Kernel::Polynomial(degree, gamma, coef) => cross_validate(
                    SVR::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef)),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Kernel::RBF(gamma) => cross_validate(
                    SVR::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_kernel(Kernels::rbf(gamma)),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Kernel::Sigmoid(gamma, coef) => cross_validate(
                    SVR::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_kernel(Kernels::sigmoid(gamma, coef)),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
            };
            let end = Instant::now();
            let d = end.duration_since(start);
            self.add_model(Algorithm::SVR, cv, d);
        }

        if !self.settings.skiplist.contains(&Algorithm::Lasso) {
            let start = Instant::now();

            let cv = cross_validate(
                Lasso::fit,
                &self.x,
                &self.y,
                self.settings.lasso_settings.as_ref().unwrap().clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();

            let end = Instant::now();
            self.add_model(Algorithm::Lasso, cv, end.duration_since(start));
        }

        if !self.settings.skiplist.contains(&Algorithm::Ridge) {
            let start = Instant::now();
            let cv = cross_validate(
                RidgeRegression::fit,
                &self.x,
                &self.y,
                self.settings.ridge_settings.as_ref().unwrap().clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            let d = end.duration_since(start);
            self.add_model(Algorithm::Ridge, cv, d);
        }

        if !self.settings.skiplist.contains(&Algorithm::ElasticNet) {
            let start = Instant::now();
            let cv = cross_validate(
                ElasticNet::fit,
                &self.x,
                &self.y,
                self.settings.elastic_net_settings.as_ref().unwrap().clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            let d = end.duration_since(start);
            self.add_model(Algorithm::ElasticNet, cv, d);
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::DecisionTreeRegressor)
        {
            let start = Instant::now();
            let cv = cross_validate(
                DecisionTreeRegressor::fit,
                &self.x,
                &self.y,
                self.settings
                    .decision_tree_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            let d = end.duration_since(start);
            self.add_model(Algorithm::DecisionTreeRegressor, cv, d);
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::RandomForestRegressor)
        {
            let start = Instant::now();
            let cv = cross_validate(
                RandomForestRegressor::fit,
                &self.x,
                &self.y,
                self.settings
                    .random_forest_regressor_settings
                    .as_ref()
                    .unwrap()
                    .clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            let d = end.duration_since(start);
            self.add_model(Algorithm::RandomForestRegressor, cv, d);
        }

        if !self.settings.skiplist.contains(&Algorithm::KNNRegressor) {
            let start = Instant::now();
            let cv = match self
                .settings
                .knn_regressor_settings
                .as_ref()
                .unwrap()
                .distance
            {
                Distance::Euclidean => cross_validate(
                    KNNRegressor::fit,
                    &self.x,
                    &self.y,
                    SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::euclidian()),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Distance::Manhattan => cross_validate(
                    KNNRegressor::fit,
                    &self.x,
                    &self.y,
                    SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::manhattan()),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Distance::Minkowski(p) => cross_validate(
                    KNNRegressor::fit,
                    &self.x,
                    &self.y,
                    SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::minkowski(p)),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Distance::Mahalanobis => cross_validate(
                    KNNRegressor::fit,
                    &self.x,
                    &self.y,
                    SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::mahalanobis(&self.x)),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
                Distance::Hamming => cross_validate(
                    KNNRegressor::fit,
                    &self.x,
                    &self.y,
                    SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::hamming()),
                    self.get_kfolds(),
                    metric,
                )
                .unwrap(),
            };
            let end = Instant::now();
            let d = end.duration_since(start);

            self.add_model(Algorithm::KNNRegressor, cv, d);
        }
    }

    /// Trains the best model found during comparison. It will panic if `compare_models` is not called first.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.compare_models();
    /// model.train_final_model();
    /// ```
    pub fn train_final_model(&mut self) {
        if self.comparison.len() == 0 {
            panic!("Please run the `compare_models` method first before trying to train a final model.");
        }
        match self.comparison[0].name {
            Algorithm::LogisticRegression => {
                self.final_model = bincode::serialize(
                    &LogisticRegression::fit(
                        &self.x,
                        &self.y,
                        self.settings.logistic_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::KNNClassifier => match self
                .settings
                .knn_classifier_settings
                .as_ref()
                .unwrap()
                .distance
            {
                Distance::Euclidean => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_weight(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_algorithm(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_distance(Distances::euclidian());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Manhattan => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_weight(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_algorithm(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_distance(Distances::manhattan());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Minkowski(p) => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_weight(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_algorithm(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_distance(Distances::minkowski(p));
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Mahalanobis => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_weight(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_algorithm(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_distance(Distances::mahalanobis(&self.x));
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Hamming => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_classifier_settings.as_ref().unwrap().k)
                        .with_weight(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_algorithm(
                            self.settings
                                .knn_classifier_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_distance(Distances::hamming());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
            },
            Algorithm::RandomForestClassifier => {
                self.final_model = bincode::serialize(
                    &RandomForestClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings
                            .random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::DecisionTreeClassifier => {
                self.final_model = bincode::serialize(
                    &DecisionTreeClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings
                            .decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::SVC => match self.settings.svc_settings.as_ref().unwrap().kernel {
                Kernel::Linear => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svc_settings.as_ref().unwrap().c)
                        .with_epoch(self.settings.svc_settings.as_ref().unwrap().epoch)
                        .with_kernel(Kernels::linear());
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Polynomial(degree, gamma, coef) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svc_settings.as_ref().unwrap().c)
                        .with_epoch(self.settings.svc_settings.as_ref().unwrap().epoch)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::RBF(gamma) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svc_settings.as_ref().unwrap().c)
                        .with_epoch(self.settings.svc_settings.as_ref().unwrap().epoch)
                        .with_kernel(Kernels::rbf(gamma));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Sigmoid(gamma, coef) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svc_settings.as_ref().unwrap().c)
                        .with_epoch(self.settings.svc_settings.as_ref().unwrap().epoch)
                        .with_kernel(Kernels::sigmoid(gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
            },

            Algorithm::GaussianNaiveBayes => {
                self.final_model = bincode::serialize(
                    &GaussianNB::fit(
                        &self.x,
                        &self.y,
                        self.settings.gaussian_nb_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }

            Algorithm::CategoricalNaiveBayes => {
                self.final_model = bincode::serialize(
                    &CategoricalNB::fit(
                        &self.x,
                        &self.y,
                        self.settings
                            .categorical_nb_settings
                            .as_ref()
                            .unwrap()
                            .clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }

            Algorithm::Linear => {
                self.final_model = bincode::serialize(
                    &LinearRegression::fit(
                        &self.x,
                        &self.y,
                        self.settings.linear_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::Lasso => {
                self.final_model = bincode::serialize(
                    &Lasso::fit(
                        &self.x,
                        &self.y,
                        self.settings.lasso_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::Ridge => {
                self.final_model = bincode::serialize(
                    &RidgeRegression::fit(
                        &self.x,
                        &self.y,
                        self.settings.ridge_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::ElasticNet => {
                self.final_model = bincode::serialize(
                    &ElasticNet::fit(
                        &self.x,
                        &self.y,
                        self.settings.elastic_net_settings.as_ref().unwrap().clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::RandomForestRegressor => {
                self.final_model = bincode::serialize(
                    &RandomForestRegressor::fit(
                        &self.x,
                        &self.y,
                        self.settings
                            .random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::KNNRegressor => match self
                .settings
                .knn_regressor_settings
                .as_ref()
                .unwrap()
                .distance
            {
                Distance::Euclidean => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::euclidian());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Manhattan => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::manhattan());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Minkowski(p) => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::minkowski(p));

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Mahalanobis => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::mahalanobis(&self.x));

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Hamming => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_regressor_settings.as_ref().unwrap().k)
                        .with_algorithm(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .unwrap()
                                .algorithm
                                .clone(),
                        )
                        .with_weight(
                            self.settings
                                .knn_regressor_settings
                                .as_ref()
                                .as_ref()
                                .unwrap()
                                .weight
                                .clone(),
                        )
                        .with_distance(Distances::hamming());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
            },
            Algorithm::SVR => match self.settings.svr_settings.as_ref().unwrap().kernel {
                Kernel::Linear => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().eps)
                        .with_kernel(Kernels::linear());
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Polynomial(degree, gamma, coef) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().eps)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::RBF(gamma) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().eps)
                        .with_kernel(Kernels::rbf(gamma));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Sigmoid(gamma, coef) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.as_ref().unwrap().tol)
                        .with_c(self.settings.svr_settings.as_ref().unwrap().c)
                        .with_eps(self.settings.svr_settings.as_ref().unwrap().eps)
                        .with_kernel(Kernels::sigmoid(gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
            },
            Algorithm::DecisionTreeRegressor => {
                self.final_model = bincode::serialize(
                    &DecisionTreeRegressor::fit(
                        &self.x,
                        &self.y,
                        self.settings
                            .decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
        }
    }

    /// Predict values using the final model based on a vec.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.compare_models();
    /// model.train_final_model();
    /// model.predict_from_vec(vec![vec![5.0; 10]; 5]);
    /// ```
    /// Note that the calls to `compare_models` and `train_final_model` can be replaced with a single call to `auto`.
    pub fn predict_from_vec(&mut self, x: Vec<Vec<f32>>) -> Vec<f32> {
        self.predict(&DenseMatrix::from_2d_vec(&x))
    }

    /// Predict values using the final model based on ndarray.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// use ndarray::arr2;
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.compare_models();
    /// model.train_final_model();
    /// model.predict_from_ndarray(
    ///     arr2(&[
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ///     ])
    /// );
    /// ```
    /// Note that the calls to `compare_models` and `train_final_model` can be replaced with a single call to `auto`.
    #[cfg_attr(docsrs, doc(cfg(feature = "nd")))]
    #[cfg(any(feature = "nd"))]
    pub fn predict_from_ndarray(&mut self, x: Array2<f32>) -> Vec<f32> {
        self.predict(&DenseMatrix::from_array(
            x.shape()[0],
            x.shape()[1],
            x.as_slice().unwrap(),
        ))
    }

    /// Runs an interactive GUI to demonstrate the final model
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.auto();
    /// model.run_gui();
    /// ```
    /// ![Example of interactive gui demo](https://raw.githubusercontent.com/cmccomb/rust-automl/master/assets/gui.png)
    #[cfg_attr(docsrs, doc(cfg(feature = "gui")))]
    #[cfg(any(feature = "gui"))]
    pub fn run_gui(self) {
        if self.final_model.len() == 0 {
            panic!()
        }
        let native_options = eframe::NativeOptions::default();
        eframe::run_native(Box::new(self), native_options);
    }
}

/// Private functions go here
impl SupervisedModel {
    fn predict(&mut self, x: &DenseMatrix<f32>) -> Vec<f32> {
        if self.final_model.len() == 0 {
            panic!("Please run the `train_final_model` method first before attempting inference with `predict`.")
        }
        self.x = self.preprocess(self.x.clone());
        match self.comparison[0].name {
            Algorithm::Linear => {
                LinearRegressionWrapper::predict(x, &self.final_model, &self.settings)
            }
            Algorithm::Lasso => {
                let model: Lasso<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::Ridge => {
                let model: RidgeRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::ElasticNet => {
                let model: ElasticNet<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::RandomForestRegressor => {
                let model: RandomForestRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::KNNRegressor => match self
                .settings
                .knn_regressor_settings
                .as_ref()
                .unwrap()
                .distance
            {
                Distance::Euclidean => {
                    let model: KNNRegressor<f32, Euclidian> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Manhattan => {
                    let model: KNNRegressor<f32, Manhattan> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Minkowski(_) => {
                    let model: KNNRegressor<f32, Minkowski> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Mahalanobis => {
                    let model: KNNRegressor<f32, Mahalanobis<f32, DenseMatrix<f32>>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Hamming => {
                    let model: KNNRegressor<f32, Hamming> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
            },
            Algorithm::SVR => match self.settings.svr_settings.as_ref().unwrap().kernel {
                Kernel::Linear => {
                    let model: SVR<f32, DenseMatrix<f32>, LinearKernel> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::Polynomial(_, _, _) => {
                    let model: SVR<f32, DenseMatrix<f32>, PolynomialKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::RBF(_) => {
                    let model: SVR<f32, DenseMatrix<f32>, RBFKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::Sigmoid(_, _) => {
                    let model: SVR<f32, DenseMatrix<f32>, SigmoidKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
            },
            Algorithm::DecisionTreeRegressor => {
                let model: DecisionTreeRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::LogisticRegression => {
                let model: LogisticRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::RandomForestClassifier => {
                let model: RandomForestClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::DecisionTreeClassifier => {
                let model: DecisionTreeClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::KNNClassifier => {
                KNNClassifierWrapper::predict(x, &self.final_model, &self.settings)
            }
            Algorithm::SVC => match self.settings.svc_settings.as_ref().unwrap().kernel {
                Kernel::Linear => {
                    let model: SVC<f32, DenseMatrix<f32>, LinearKernel> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::Polynomial(_, _, _) => {
                    let model: SVC<f32, DenseMatrix<f32>, PolynomialKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::RBF(_) => {
                    let model: SVC<f32, DenseMatrix<f32>, RBFKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Kernel::Sigmoid(_, _) => {
                    let model: SVC<f32, DenseMatrix<f32>, SigmoidKernel<f32>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
            },
            Algorithm::GaussianNaiveBayes => {
                let model: GaussianNB<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::CategoricalNaiveBayes => {
                let model: CategoricalNB<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }

    fn interaction_features(mut x: DenseMatrix<f32>) -> DenseMatrix<f32> {
        let (_, width) = x.shape();
        for i in 0..width {
            for j in (i + 1)..width {
                let feature = elementwise_multiply(&x.get_col_as_vec(i), &x.get_col_as_vec(j));
                let new_column = DenseMatrix::from_row_vector(feature).transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    fn polynomial_features(mut x: DenseMatrix<f32>, order: usize) -> DenseMatrix<f32> {
        let (height, width) = x.shape();
        for n in 2..=order {
            let combinations = (0..width).into_iter().combinations_with_replacement(n);
            for combo in combinations {
                let mut feature = vec![1.0; height];
                for column in combo {
                    feature = elementwise_multiply(&x.get_col_as_vec(column), &feature);
                }
                let new_column = DenseMatrix::from_row_vector(feature).transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    fn pca_features(&mut self, x: DenseMatrix<f32>, n: usize) -> DenseMatrix<f32> {
        if let None = self.preprocessing.0 {
            let pca = PCA::fit(
                &x,
                PCAParameters::default()
                    .with_n_components(n)
                    .with_use_correlation_matrix(true),
            )
            .unwrap();
            self.preprocessing.0 = Some(pca);
        }
        self.preprocessing
            .0
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    fn svd_features(&mut self, x: DenseMatrix<f32>, n: usize) -> DenseMatrix<f32> {
        if let None = self.preprocessing.1 {
            let svd = SVD::fit(&x, SVDParameters::default().with_n_components(n)).unwrap();
            self.preprocessing.1 = Some(svd);
        }
        self.preprocessing
            .1
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    fn preprocess(&mut self, x: DenseMatrix<f32>) -> DenseMatrix<f32> {
        match self.settings.preprocessing {
            PreProcessing::None => x,
            PreProcessing::AddInteractions => SupervisedModel::interaction_features(x),
            PreProcessing::AddPolynomial { order } => {
                SupervisedModel::polynomial_features(x, order)
            }
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => self.pca_features(x, number_of_components),
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => self.svd_features(x, number_of_components),
        }
    }

    fn count_classes(y: &Vec<f32>) -> usize {
        let mut sorted_targets = y.clone();
        sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
        sorted_targets.dedup();
        sorted_targets.len()
    }

    fn add_model(
        &mut self,
        name: Algorithm,
        score: CrossValidationResult<f32>,
        duration: Duration,
    ) {
        self.comparison.push(Model {
            score,
            name,
            duration,
        });
        self.sort();
    }

    fn record_model(&mut self, model: (CrossValidationResult<f32>, Algorithm, Duration)) {
        self.comparison.push(Model {
            score: model.0,
            name: model.1,
            duration: model.2,
        });
        self.sort();
    }

    fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.settings.number_of_folds)
            .with_shuffle(self.settings.shuffle)
    }

    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            a.score
                .mean_test_score()
                .partial_cmp(&b.score.mean_test_score())
                .unwrap_or(Equal)
        });
        if self.settings.sort_by == Metric::RSquared || self.settings.sort_by == Metric::Accuracy {
            self.comparison.reverse();
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "display")))]
#[cfg(any(feature = "display"))]
impl Display for SupervisedModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec![
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new("Time").add_attribute(Attribute::Bold),
            Cell::new(format!("Training {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
        ]);
        for model in &self.comparison {
            let mut row_vec = vec![];
            row_vec.push(format!("{}", &model.name));
            row_vec.push(format!("{}", format_duration(model.duration)));
            let decider =
                ((model.score.mean_train_score() + model.score.mean_test_score()) / 2.0).abs();
            if decider > 0.01 && decider < 1000.0 {
                row_vec.push(format!("{:.2}", &model.score.mean_train_score()));
                row_vec.push(format!("{:.2}", &model.score.mean_test_score()));
            } else {
                row_vec.push(format!("{:.3e}", &model.score.mean_train_score()));
                row_vec.push(format!("{:.3e}", &model.score.mean_test_score()));
            }

            table.add_row(row_vec);
        }
        write!(f, "{}\n", table)
    }
}

/// Settings for supervised models
pub struct Settings {
    sort_by: Metric,
    model_type: ModelType,
    skiplist: Vec<Algorithm>,
    number_of_folds: usize,
    shuffle: bool,
    verbose: bool,
    preprocessing: PreProcessing,
    linear_settings: Option<LinearRegressionParameters>,
    svr_settings: Option<SVRParameters>,
    lasso_settings: Option<LassoParameters<f32>>,
    ridge_settings: Option<RidgeRegressionParameters<f32>>,
    elastic_net_settings: Option<ElasticNetParameters<f32>>,
    decision_tree_regressor_settings: Option<DecisionTreeRegressorParameters>,
    random_forest_regressor_settings: Option<RandomForestRegressorParameters>,
    knn_regressor_settings: Option<KNNRegressorParameters>,
    logistic_settings: Option<LogisticRegressionParameters>,
    random_forest_classifier_settings: Option<RandomForestClassifierParameters>,
    knn_classifier_settings: Option<KNNClassifierParameters>,
    svc_settings: Option<SVCParameters>,
    decision_tree_classifier_settings: Option<DecisionTreeClassifierParameters>,
    gaussian_nb_settings: Option<GaussianNBParameters<f32>>,
    categorical_nb_settings: Option<CategoricalNBParameters<f32>>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            model_type: ModelType::None,
            skiplist: vec![
                Algorithm::LogisticRegression,
                Algorithm::RandomForestClassifier,
                Algorithm::KNNClassifier,
                Algorithm::SVC,
                Algorithm::DecisionTreeClassifier,
                Algorithm::CategoricalNaiveBayes,
                Algorithm::GaussianNaiveBayes,
                Algorithm::Linear,
                Algorithm::Lasso,
                Algorithm::Ridge,
                Algorithm::ElasticNet,
                Algorithm::SVR,
                Algorithm::DecisionTreeRegressor,
                Algorithm::RandomForestRegressor,
                Algorithm::KNNRegressor,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: None,
            svr_settings: None,
            lasso_settings: None,
            ridge_settings: None,
            elastic_net_settings: None,
            decision_tree_regressor_settings: None,
            random_forest_regressor_settings: None,
            knn_regressor_settings: None,
            logistic_settings: None,
            random_forest_classifier_settings: None,
            knn_classifier_settings: None,
            svc_settings: None,
            decision_tree_classifier_settings: None,
            gaussian_nb_settings: None,
            categorical_nb_settings: None,
        }
    }
}

impl Settings {
    pub(crate) fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.number_of_folds)
            .with_shuffle(self.shuffle)
    }

    pub(crate) fn get_metric(&self) -> Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32> {
        Box::new(match self.sort_by {
            Metric::RSquared => r2,
            Metric::MeanAbsoluteError => mean_absolute_error,
            Metric::MeanSquaredError => mean_squared_error,
            Metric::Accuracy => accuracy,
            Metric::None => panic!("A metric must be set."),
        })
    }

    /// Creates default settings for regression
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default_regression();
    /// ```
    pub fn default_regression() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            model_type: ModelType::Regression,
            skiplist: vec![
                Algorithm::LogisticRegression,
                Algorithm::RandomForestClassifier,
                Algorithm::KNNClassifier,
                Algorithm::SVC,
                Algorithm::DecisionTreeClassifier,
                Algorithm::CategoricalNaiveBayes,
                Algorithm::GaussianNaiveBayes,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: Some(LinearRegressionParameters::default()),
            svr_settings: Some(SVRParameters::default()),
            lasso_settings: Some(LassoParameters::default()),
            ridge_settings: Some(RidgeRegressionParameters::default()),
            elastic_net_settings: Some(ElasticNetParameters::default()),
            decision_tree_regressor_settings: Some(DecisionTreeRegressorParameters::default()),
            random_forest_regressor_settings: Some(RandomForestRegressorParameters::default()),
            knn_regressor_settings: Some(KNNRegressorParameters::default()),
            logistic_settings: None,
            random_forest_classifier_settings: None,
            knn_classifier_settings: None,
            svc_settings: None,
            decision_tree_classifier_settings: None,
            gaussian_nb_settings: None,
            categorical_nb_settings: None,
        }
    }

    /// Creates default settings for classification
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default_classification();
    /// ```
    pub fn default_classification() -> Self {
        Settings {
            sort_by: Metric::Accuracy,
            model_type: ModelType::Classification,
            skiplist: vec![
                Algorithm::Linear,
                Algorithm::Lasso,
                Algorithm::Ridge,
                Algorithm::ElasticNet,
                Algorithm::SVR,
                Algorithm::DecisionTreeRegressor,
                Algorithm::RandomForestRegressor,
                Algorithm::KNNRegressor,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: None,
            svr_settings: None,
            lasso_settings: None,
            ridge_settings: None,
            elastic_net_settings: None,
            decision_tree_regressor_settings: None,
            random_forest_regressor_settings: None,
            knn_regressor_settings: None,
            logistic_settings: Some(LogisticRegressionParameters::default()),
            random_forest_classifier_settings: Some(RandomForestClassifierParameters::default()),
            knn_classifier_settings: Some(KNNClassifierParameters::default()),
            svc_settings: Some(SVCParameters::default()),
            decision_tree_classifier_settings: Some(DecisionTreeClassifierParameters::default()),
            gaussian_nb_settings: Some(GaussianNBParameters::default()),
            categorical_nb_settings: Some(CategoricalNBParameters::default()),
        }
    }

    /// Specify number of folds for cross-validation
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().with_number_of_folds(3);
    /// ```
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().shuffle_data(true);
    /// ```
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify whether or not to be verbose
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().verbose(true);
    /// ```
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Specify what type of preprocessing should be performed
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::PreProcessing;
    /// let settings = Settings::default().with_preprocessing(PreProcessing::AddInteractions);
    /// ```
    pub fn with_preprocessing(mut self, pre: PreProcessing) -> Self {
        self.preprocessing = pre;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::Algorithm;
    /// let settings = Settings::default().skip(Algorithm::RandomForestRegressor);
    /// ```
    pub fn skip(mut self, skip: Algorithm) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Adds a specific sorting function to the settings
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::Metric;
    /// let settings = Settings::default().sorted_by(Metric::RSquared);
    /// ```
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for random_forest
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::RandomForestClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_classifier_settings(RandomForestClassifierParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_classifier_settings(
        mut self,
        settings: RandomForestClassifierParameters,
    ) -> Self {
        self.random_forest_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::LogisticRegressionParameters;
    /// let settings = Settings::default()
    ///     .with_logistic_settings(LogisticRegressionParameters::default());
    /// ```
    pub fn with_logistic_settings(mut self, settings: LogisticRegressionParameters) -> Self {
        self.logistic_settings = Some(settings);
        self
    }

    /// Specify settings for support vector classifier
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{SVCParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svc_settings(SVCParameters::default()
    ///         .with_epoch(10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svc_settings(mut self, settings: SVCParameters) -> Self {
        self.svc_settings = Some(settings);
        self
    }

    /// Specify settings for decision tree classifier
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::DecisionTreeClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_classifier_settings(DecisionTreeClassifierParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_classifier_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{KNNClassifierParameters,
    ///     KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_classifier_settings(KNNClassifierParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_classifier_settings(mut self, settings: KNNClassifierParameters) -> Self {
        self.knn_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for Gaussian Naive Bayes
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::GaussianNBParameters;
    /// let settings = Settings::default()
    ///     .with_gaussian_nb_settings(GaussianNBParameters::default()
    ///         .with_priors(vec![1.0, 1.0])
    ///     );
    /// ```
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters<f32>) -> Self {
        self.gaussian_nb_settings = Some(settings);
        self
    }

    /// Specify settings for Categorical Naive Bayes
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::CategoricalNBParameters;
    /// let settings = Settings::default()
    ///     .with_categorical_nb_settings(CategoricalNBParameters::default()
    ///         .with_alpha(1.0)
    ///     );
    /// ```
    pub fn with_categorical_nb_settings(mut self, settings: CategoricalNBParameters<f32>) -> Self {
        self.categorical_nb_settings = Some(settings);
        self
    }

    /// Specify settings for linear regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{LinearRegressionParameters, LinearRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_linear_settings(LinearRegressionParameters::default()
    ///         .with_solver(LinearRegressionSolverName::QR)
    ///     );
    /// ```
    pub fn with_linear_settings(mut self, settings: LinearRegressionParameters) -> Self {
        self.linear_settings = Some(settings);
        self
    }

    /// Specify settings for lasso regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::LassoParameters;
    /// let settings = Settings::default()
    ///     .with_lasso_settings(LassoParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_tol(1e-10)
    ///         .with_normalize(true)
    ///         .with_max_iter(10_000)
    ///     );
    /// ```
    pub fn with_lasso_settings(mut self, settings: LassoParameters<f32>) -> Self {
        self.lasso_settings = Some(settings);
        self
    }

    /// Specify settings for ridge regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{RidgeRegressionParameters, RidgeRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_ridge_settings(RidgeRegressionParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_normalize(true)
    ///         .with_solver(RidgeRegressionSolverName::Cholesky)
    ///     );
    /// ```
    pub fn with_ridge_settings(mut self, settings: RidgeRegressionParameters<f32>) -> Self {
        self.ridge_settings = Some(settings);
        self
    }

    /// Specify settings for elastic net
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::ElasticNetParameters;
    /// let settings = Settings::default()
    ///     .with_elastic_net_settings(ElasticNetParameters::default()
    ///         .with_tol(1e-10)
    ///         .with_normalize(true)
    ///         .with_alpha(1.0)
    ///         .with_max_iter(10_000)
    ///         .with_l1_ratio(0.5)    
    ///     );
    /// ```
    pub fn with_elastic_net_settings(mut self, settings: ElasticNetParameters<f32>) -> Self {
        self.elastic_net_settings = Some(settings);
        self
    }

    /// Specify settings for KNN regressor
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{KNNRegressorParameters,
    ///     KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_regressor_settings(KNNRegressorParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_regressor_settings(mut self, settings: KNNRegressorParameters) -> Self {
        self.knn_regressor_settings = Some(settings);
        self
    }

    /// Specify settings for support vector regressor
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{SVRParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svr_settings(SVRParameters::default()
    ///         .with_eps(1e-10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svr_settings(mut self, settings: SVRParameters) -> Self {
        self.svr_settings = Some(settings);
        self
    }

    /// Specify settings for random forest
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::RandomForestRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_regressor_settings(RandomForestRegressorParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_regressor_settings(
        mut self,
        settings: RandomForestRegressorParameters,
    ) -> Self {
        self.random_forest_regressor_settings = Some(settings);
        self
    }

    /// Specify settings for decision tree
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::DecisionTreeRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_regressor_settings(DecisionTreeRegressorParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_regressor_settings(
        mut self,
        settings: DecisionTreeRegressorParameters,
    ) -> Self {
        self.decision_tree_regressor_settings = Some(settings);
        self
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "display")))]
#[cfg(any(feature = "display"))]
impl Display for Settings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Prep new table
        let mut table = Table::new();

        // Get list of algorithms to skip
        let mut skiplist = String::new();
        if self.skiplist.len() == 0 {
            skiplist.push_str("None ");
        } else {
            for algorithm_to_skip in &self.skiplist {
                skiplist.push_str(&*format!("{}\n", algorithm_to_skip));
            }
        }

        // Build out the table
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_SOLID_INNER_BORDERS)
            .set_header(vec![
                Cell::new("Settings").add_attribute(Attribute::Bold),
                Cell::new("Value").add_attribute(Attribute::Bold),
            ])
            .add_row(vec![Cell::new("General").add_attribute(Attribute::Italic)])
            .add_row(vec!["    Model Type", &*format!("{}", self.model_type)])
            .add_row(vec!["    Verbose", &*format!("{}", self.verbose)])
            .add_row(vec!["    Sorting Metric", &*format!("{}", self.sort_by)])
            .add_row(vec!["    Shuffle Data", &*format!("{}", self.shuffle)])
            .add_row(vec![
                "    Number of CV Folds",
                &*format!("{}", self.number_of_folds),
            ])
            .add_row(vec![
                "    Pre-Processing",
                &*format!("{}", self.preprocessing),
            ])
            .add_row(vec![
                "    Skipped Algorithms",
                &*format!("{}", &skiplist[0..skiplist.len() - 1]),
            ]);
        if !self.skiplist.contains(&Algorithm::Linear) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Linear).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Solver",
                    match self.linear_settings.as_ref().unwrap().solver {
                        LinearRegressionSolverName::QR => "QR",
                        LinearRegressionSolverName::SVD => "SVD",
                    },
                ]);
        }
        if !self.skiplist.contains(&Algorithm::Ridge) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Ridge).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Solver",
                    match self.ridge_settings.as_ref().unwrap().solver {
                        RidgeRegressionSolverName::Cholesky => "Cholesky",
                        RidgeRegressionSolverName::SVD => "SVD",
                    },
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.ridge_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.ridge_settings.as_ref().unwrap().normalize),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::Lasso) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Lasso).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().tol),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::ElasticNet) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::ElasticNet).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    L1 Ratio",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().l1_ratio),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::DecisionTreeRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::DecisionTreeRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::RandomForestRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::RandomForestRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .n_trees
                    ),
                ])
                .add_row(vec![
                    "    Number of split candidates",
                    &*print_option(self.random_forest_regressor_settings.as_ref().unwrap().m),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::KNNRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::KNNRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Number of neighbors",
                    &*format!("{}", self.knn_regressor_settings.as_ref().unwrap().k),
                ])
                .add_row(vec![
                    "    Search algorithm",
                    &*format!(
                        "{}",
                        print_knn_search_algorithm(
                            &self.knn_regressor_settings.as_ref().unwrap().algorithm
                        )
                    ),
                ])
                .add_row(vec![
                    "    Weighting function",
                    &*format!(
                        "{}",
                        print_knn_weight_function(
                            &self.knn_regressor_settings.as_ref().unwrap().weight
                        )
                    ),
                ])
                .add_row(vec![
                    "    Distance function",
                    &*format!(
                        "{}",
                        &self.knn_regressor_settings.as_ref().unwrap().distance
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::SVR) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::SVR).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Regularization parameter",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().c),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    Epsilon",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().eps),
                ])
                .add_row(vec![
                    "    Kernel",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().kernel),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::LogisticRegression) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::LogisticRegression).add_attribute(Attribute::Italic)
                ])
                .add_row(vec!["    N/A", "N/A"]);
        }

        if !self.skiplist.contains(&Algorithm::RandomForestClassifier) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::RandomForestClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Split Criterion",
                    match self
                        .random_forest_classifier_settings
                        .as_ref()
                        .unwrap()
                        .criterion
                    {
                        SplitCriterion::Gini => "Gini",
                        SplitCriterion::Entropy => "Entropy",
                        SplitCriterion::ClassificationError => "Classification Error",
                    },
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .n_trees
                    ),
                ])
                .add_row(vec![
                    "    Number of split candidates",
                    &*print_option(self.random_forest_classifier_settings.as_ref().unwrap().m),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::KNNClassifier) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::KNNClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Number of neighbors",
                    &*format!("{}", self.knn_classifier_settings.as_ref().unwrap().k),
                ])
                .add_row(vec![
                    "    Search algorithm",
                    &*format!(
                        "{}",
                        print_knn_search_algorithm(
                            &self.knn_classifier_settings.as_ref().unwrap().algorithm
                        )
                    ),
                ])
                .add_row(vec![
                    "    Weighting function",
                    &*format!(
                        "{}",
                        print_knn_weight_function(
                            &self.knn_classifier_settings.as_ref().unwrap().weight
                        )
                    ),
                ])
                .add_row(vec![
                    "    Distance function",
                    &*format!(
                        "{}",
                        &self.knn_classifier_settings.as_ref().unwrap().distance
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::SVC) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::SVC).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Regularization parameter",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().c),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    Epoch",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().epoch),
                ])
                .add_row(vec![
                    "    Kernel",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().kernel),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::DecisionTreeClassifier) {
            table
                .add_row(vec![
                    "    Split Criterion",
                    match self
                        .random_forest_classifier_settings
                        .as_ref()
                        .unwrap()
                        .criterion
                    {
                        SplitCriterion::Gini => "Gini",
                        SplitCriterion::Entropy => "Entropy",
                        SplitCriterion::ClassificationError => "Classification Error",
                    },
                ])
                .add_row(vec![
                    Cell::new(Algorithm::DecisionTreeClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::CategoricalNaiveBayes) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::CategoricalNaiveBayes).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Smoothing parameter",
                    &*format!("{}", self.categorical_nb_settings.as_ref().unwrap().alpha),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::GaussianNaiveBayes) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::GaussianNaiveBayes).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Priors",
                    &*debug_option(self.gaussian_nb_settings.as_ref().unwrap().clone().priors),
                ]);
        }

        write!(f, "{}\n", table)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "gui")))]
#[cfg(any(feature = "gui"))]
impl epi::App for SupervisedModel {
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Add a heading that displays the type of model this is
            ui.heading(format!("{}", self.comparison[0].name));

            // Add a label that shows the prediction
            ui.label(format!(
                "Prediction: y = {}",
                self.predict(&DenseMatrix::from_2d_vec(&vec![self.current_x.to_vec(); 1]))[0]
            ));

            // Separating the model name and prediction from the input values
            ui.separator();

            // Step through input values to make sliders
            for i in 0..self.current_x.len() {
                // Figure out the maximum in the training dataa
                let maxx = self
                    .x
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::max);

                // Figure out the minimum in the training data
                let minn = self
                    .x
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::min);

                // Add the slider
                ui.add(
                    egui::Slider::new(&mut self.current_x[i], minn..=maxx).text(format!("x_{}", i)),
                );
            }
        });
    }

    fn name(&self) -> &str {
        "Model Demo"
    }
}

/// This contains the results of a single model
struct Model {
    score: CrossValidationResult<f32>,
    name: Algorithm,
    duration: Duration,
}

enum ModelType {
    None,
    Regression,
    Classification,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::None => write!(f, "None"),
            ModelType::Regression => write!(f, "Regression"),
            ModelType::Classification => write!(f, "Classification"),
        }
    }
}
