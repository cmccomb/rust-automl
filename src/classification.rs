//! # Auto-ML for Classification
//! This module provides the ability to quickly train and compare a variety of regression models. For instance, this example:
//! ```
//! use automl::classification::Classifier;
//! let mut classifier = Classifier::default();
//! classifier.with_dataset(smartcore::dataset::breast_cancer::load_dataset());
//! classifier.compare_models();
//! ```
//! Will output the following comparison table:
//! ```text
//! ┌────────────────────────────────┬─────────────────────┬───────────────────┬──────────────────┐
//! │ Model                          │ Time                │ Training Accuracy │ Testing Accuracy │
//! ╞════════════════════════════════╪═════════════════════╪═══════════════════╪══════════════════╡
//! │ Random Forest Classifier       │ 835ms 393us 583ns   │ 1.00              │ 0.96             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ Logistic Regression Classifier │ 620ms 714us 583ns   │ 0.97              │ 0.95             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ Gaussian Naive Bayes           │ 6ms 529us           │ 0.94              │ 0.93             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ Categorical Naive Bayes        │ 2ms 922us 250ns     │ 0.96              │ 0.93             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ Decision Tree Classifier       │ 15ms 404us 750ns    │ 1.00              │ 0.93             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ KNN Classifier                 │ 28ms 874us 208ns    │ 0.96              │ 0.92             │
//! ├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
//! │ Support Vector Classifier      │ 4s 187ms 61us 708ns │ 0.57              │ 0.57             │
//! └────────────────────────────────┴─────────────────────┴───────────────────┴──────────────────┘
//! ```
//! To learn more about how to customize the settings for the individual models, refer to
//! the [settings module](settings).

pub mod settings;
use settings::{
    Algorithm, CategoricalNBParameters, DecisionTreeClassifierParameters, Distance,
    GaussianNBParameters, KNNClassifierParameters, Kernel, LogisticRegressionParameters, Metric,
    RandomForestClassifierParameters, SVCParameters,
};

use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};
use polars::prelude::{CsvReader, DataFrame, Float32Type, SerReader};
use smartcore::math::distance::{
    euclidian::Euclidian, hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan,
    minkowski::Minkowski, Distances,
};
use smartcore::svm::{Kernels, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_classifier::RandomForestClassifier,
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::LogisticRegression,
    metrics::accuracy,
    model_selection::{cross_validate, CrossValidationResult, KFold},
    naive_bayes::{categorical::CategoricalNB, gaussian::GaussianNB},
    neighbors::knn_classifier::{
        KNNClassifier, KNNClassifierParameters as SmartcoreKNNClassifierParameters,
    },
    svm::svc::{SVCParameters as SmartcoreSVCParameters, SVC},
    tree::decision_tree_classifier::DecisionTreeClassifier,
};

use humantime::format_duration;
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
};

use eframe::{egui, epi};
use smartcore::linalg::BaseMatrix;

/// Trains and compares classification models
pub struct Classifier {
    settings: Settings,
    pub(crate) x: DenseMatrix<f32>,
    y: Vec<f32>,
    comparison: Vec<Model>,
    final_model: Vec<u8>,
    number_of_classes: usize,
    current_x: Vec<f32>,
}

impl Classifier {
    /// Establish a new classifier with settings
    pub fn new(x: DenseMatrix<f32>, y: Vec<f32>, settings: Settings) -> Self {
        Self {
            settings,
            x: x.clone(),
            y,
            comparison: vec![],
            final_model: vec![],
            number_of_classes: 0,
            current_x: vec![0.0; x.shape().1],
        }
    }

    /// Add settings to the classifier
    pub fn with_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    /// Runs a model comparison and trains a final model. [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(&mut self) {
        self.compare_models();
        self.train_final_model();
    }

    /// Add data to regressor object
    pub fn with_data_from_vec(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>) {
        self.x = DenseMatrix::from_2d_vec(&x);
        self.y = y;
        self.count_classes();
        self.current_x = vec![0.0; self.x.shape().1];
    }

    /// Add data to regressor object
    pub fn with_data_from_ndarray(&mut self, x: Array2<f32>, y: Array1<f32>) {
        self.x = DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap());
        self.y = y.to_vec();
        self.count_classes();
        self.current_x = vec![0.0; self.x.shape().1];
    }

    /// Add a dataset to regressor object
    pub fn with_dataset(&mut self, dataset: Dataset<f32, f32>) {
        self.x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        self.y = dataset.target;
        self.count_classes();
        self.current_x = vec![0.0; self.x.shape().1];
    }

    /// Add data from a csv
    pub fn with_data_from_csv(&mut self, filepath: &str, target: usize, header: bool) {
        let df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

        // Take a look at the data
        if self.settings.verbose {
            println!("{}", df);
        }

        // Get target variables
        let target_column_name = df.get_column_names()[target];
        let series = df.column(target_column_name).unwrap().clone();
        let target_df = DataFrame::new(vec![series]).unwrap();
        let ndarray = target_df.to_ndarray::<Float32Type>().unwrap();
        self.y = ndarray.into_raw_vec();

        // Get the rest of the data
        let features = df.drop(target_column_name).unwrap();
        let (height, width) = features.shape();
        let ndarray = features.to_ndarray::<Float32Type>().unwrap();
        self.x = DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap());
    }

    /// This function compares all of the classification models available in the package.
    pub fn compare_models(&mut self) {
        let metric = match self.settings.sort_by {
            Metric::Accuracy => accuracy,
        };
        if !self
            .settings
            .skiplist
            .contains(&Algorithm::LogisticRegression)
        {
            let start = Instant::now();
            let cv = cross_validate(
                LogisticRegression::fit,
                &self.x,
                &self.y,
                self.settings.logistic_settings.clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            self.add_model(Algorithm::LogisticRegression, cv, end.duration_since(start));
        }

        if !self.settings.skiplist.contains(&Algorithm::RandomForest) {
            let start = Instant::now();
            let cv = cross_validate(
                RandomForestClassifier::fit,
                &self.x,
                &self.y,
                self.settings.random_forest_settings.clone(),
                self.get_kfolds(),
                metric,
            )
            .unwrap();
            let end = Instant::now();
            self.add_model(Algorithm::RandomForest, cv, end.duration_since(start));
        }

        if !self.settings.skiplist.contains(&Algorithm::KNN) {
            match self.settings.knn_settings.distance {
                Distance::Euclidean => {
                    let start = Instant::now();
                    let cv = cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNClassifierParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_distance(Distances::euclidian()),
                        self.get_kfolds(),
                        metric,
                    )
                    .unwrap();
                    let end = Instant::now();
                    self.add_model(Algorithm::KNN, cv, end.duration_since(start));
                }
                Distance::Manhattan => {
                    let start = Instant::now();
                    let cv = cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNClassifierParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_distance(Distances::manhattan()),
                        self.get_kfolds(),
                        metric,
                    )
                    .unwrap();
                    let end = Instant::now();
                    self.add_model(Algorithm::KNN, cv, end.duration_since(start));
                }
                Distance::Minkowski(p) => {
                    let start = Instant::now();
                    let cv = cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNClassifierParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_distance(Distances::minkowski(p)),
                        self.get_kfolds(),
                        metric,
                    )
                    .unwrap();
                    let end = Instant::now();
                    self.add_model(Algorithm::KNN, cv, end.duration_since(start));
                }
                Distance::Mahalanobis => {
                    let start = Instant::now();
                    let cv = cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNClassifierParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_distance(Distances::mahalanobis(&self.x)),
                        self.get_kfolds(),
                        metric,
                    )
                    .unwrap();
                    let end = Instant::now();
                    self.add_model(Algorithm::KNN, cv, end.duration_since(start));
                }
                Distance::Hamming => {
                    let start = Instant::now();
                    let cv = cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNClassifierParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_distance(Distances::hamming()),
                        self.get_kfolds(),
                        metric,
                    )
                    .unwrap();
                    let end = Instant::now();
                    self.add_model(Algorithm::KNN, cv, end.duration_since(start));
                }
            }
        }

        if !self.settings.skiplist.contains(&Algorithm::DecisionTree) {
            let start = Instant::now();
            let cv = cross_validate(
                DecisionTreeClassifier::fit,
                &self.x,
                &self.y,
                self.settings.decision_tree_settings.clone(),
                self.get_kfolds(),
                match self.settings.sort_by {
                    Metric::Accuracy => accuracy,
                },
            )
            .unwrap();
            let end = Instant::now();
            self.add_model(Algorithm::DecisionTree, cv, end.duration_since(start));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::GaussianNaiveBayes)
        {
            let start = Instant::now();
            let cv = cross_validate(
                GaussianNB::fit,
                &self.x,
                &self.y,
                self.settings.gaussian_nb_settings.clone(),
                self.get_kfolds(),
                match self.settings.sort_by {
                    Metric::Accuracy => accuracy,
                },
            )
            .unwrap();
            let end = Instant::now();
            self.add_model(Algorithm::GaussianNaiveBayes, cv, end.duration_since(start));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::CategoricalNaiveBayes)
        {
            let start = Instant::now();
            let cv = cross_validate(
                CategoricalNB::fit,
                &self.x,
                &self.y,
                self.settings.categorical_nb_settings.clone(),
                self.get_kfolds(),
                match self.settings.sort_by {
                    Metric::Accuracy => accuracy,
                },
            )
            .unwrap();
            let end = Instant::now();
            self.add_model(
                Algorithm::CategoricalNaiveBayes,
                cv,
                end.duration_since(start),
            );
        }

        if self.number_of_classes == 2 && !self.settings.skiplist.contains(&Algorithm::SVC) {
            let start = Instant::now();

            let cv = match self.settings.svc_settings.kernel {
                Kernel::Linear => cross_validate(
                    SVC::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::linear()),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::Accuracy => accuracy,
                    },
                )
                .unwrap(),
                Kernel::Polynomial(degree, gamma, coef) => cross_validate(
                    SVC::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef)),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::Accuracy => accuracy,
                    },
                )
                .unwrap(),
                Kernel::RBF(gamma) => cross_validate(
                    SVC::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::rbf(gamma)),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::Accuracy => accuracy,
                    },
                )
                .unwrap(),
                Kernel::Sigmoid(gamma, coef) => cross_validate(
                    SVC::fit,
                    &self.x,
                    &self.y,
                    SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::sigmoid(gamma, coef)),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::Accuracy => accuracy,
                    },
                )
                .unwrap(),
            };
            let end = Instant::now();
            self.add_model(Algorithm::SVC, cv, end.duration_since(start));
        };
    }

    /// Trains the best model found during comparison
    pub fn train_final_model(&mut self) {
        match self.comparison[0].name {
            Algorithm::LogisticRegression => {
                self.final_model = bincode::serialize(
                    &LogisticRegression::fit(
                        &self.x,
                        &self.y,
                        self.settings.logistic_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::KNN => match self.settings.knn_settings.distance {
                Distance::Euclidean => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_distance(Distances::euclidian());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Manhattan => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_distance(Distances::manhattan());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Minkowski(p) => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_distance(Distances::minkowski(p));
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Mahalanobis => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_distance(Distances::mahalanobis(&self.x));
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Hamming => {
                    let params = SmartcoreKNNClassifierParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_distance(Distances::hamming());
                    self.final_model =
                        bincode::serialize(&KNNClassifier::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
            },
            Algorithm::RandomForest => {
                self.final_model = bincode::serialize(
                    &RandomForestClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings.random_forest_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::DecisionTree => {
                self.final_model = bincode::serialize(
                    &DecisionTreeClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings.decision_tree_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::SVC => match self.settings.svc_settings.kernel {
                Kernel::Linear => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::linear());
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Polynomial(degree, gamma, coef) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::RBF(gamma) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::rbf(gamma));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Sigmoid(gamma, coef) => {
                    let params = SmartcoreSVCParameters::default()
                        .with_tol(self.settings.svc_settings.tol)
                        .with_c(self.settings.svc_settings.c)
                        .with_epoch(self.settings.svc_settings.epoch)
                        .with_kernel(Kernels::sigmoid(gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVC::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
            },

            Algorithm::GaussianNaiveBayes => {
                self.final_model = bincode::serialize(
                    &GaussianNB::fit(&self.x, &self.y, self.settings.gaussian_nb_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }

            Algorithm::CategoricalNaiveBayes => {
                self.final_model = bincode::serialize(
                    &CategoricalNB::fit(
                        &self.x,
                        &self.y,
                        self.settings.categorical_nb_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
        }
    }

    /// Predict values using the best model
    pub fn predict(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.comparison[0].name {
            Algorithm::LogisticRegression => {
                let model: LogisticRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::RandomForest => {
                let model: RandomForestClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::DecisionTree => {
                let model: DecisionTreeClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::KNN => match self.settings.knn_settings.distance {
                Distance::Euclidean => {
                    let model: KNNClassifier<f32, Euclidian> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Manhattan => {
                    let model: KNNClassifier<f32, Manhattan> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Minkowski(_) => {
                    let model: KNNClassifier<f32, Minkowski> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Mahalanobis => {
                    let model: KNNClassifier<f32, Mahalanobis<f32, DenseMatrix<f32>>> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
                Distance::Hamming => {
                    let model: KNNClassifier<f32, Hamming> =
                        bincode::deserialize(&*self.final_model).unwrap();
                    model.predict(x).unwrap()
                }
            },
            Algorithm::SVC => match self.settings.svc_settings.kernel {
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
}

/// Private functions go here
impl Classifier {
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

        if self.settings.verbose {
            print!("{esc}c", esc = 27 as char);
            println!("{}", self);
        }
    }

    fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.settings.number_of_folds)
            .with_shuffle(self.settings.shuffle)
    }

    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            b.score
                .mean_test_score()
                .partial_cmp(&a.score.mean_test_score())
                .unwrap_or(Equal)
        });
    }

    fn count_classes(&mut self) {
        let mut sorted_targets = self.y.clone();
        sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
        sorted_targets.dedup();
        self.number_of_classes = sorted_targets.len();
    }
}

impl Display for Classifier {
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

impl Default for Classifier {
    fn default() -> Self {
        Self {
            settings: Default::default(),
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: vec![],
            final_model: vec![],
            number_of_classes: 0,
            current_x: vec![],
        }
    }
}

/// This contains the results of a single model
struct Model {
    score: CrossValidationResult<f32>,
    name: Algorithm,
    duration: Duration,
}

/// Settings for classification algorithms and comparisons
pub struct Settings {
    skiplist: Vec<Algorithm>,
    sort_by: Metric,
    number_of_folds: usize,
    shuffle: bool,
    verbose: bool,
    logistic_settings: LogisticRegressionParameters,
    random_forest_settings: RandomForestClassifierParameters,
    knn_settings: KNNClassifierParameters,
    svc_settings: SVCParameters,
    decision_tree_settings: DecisionTreeClassifierParameters,
    gaussian_nb_settings: GaussianNBParameters<f32>,
    categorical_nb_settings: CategoricalNBParameters<f32>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            skiplist: vec![],
            sort_by: Metric::Accuracy,
            shuffle: true,
            verbose: true,
            logistic_settings: LogisticRegressionParameters::default(),
            random_forest_settings: RandomForestClassifierParameters::default(),
            knn_settings: KNNClassifierParameters::default(),
            svc_settings: SVCParameters::default(),
            decision_tree_settings: DecisionTreeClassifierParameters::default(),
            gaussian_nb_settings: GaussianNBParameters::default(),
            categorical_nb_settings: CategoricalNBParameters::default(),
            number_of_folds: 10,
        }
    }
}

impl Settings {
    /// Specify number of folds for cross-validation
    /// ```
    /// # use automl::classification::Settings;
    /// let settings = Settings::default().with_number_of_folds(3);
    /// ```
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    /// ```
    /// # use automl::classification::Settings;
    /// let settings = Settings::default().shuffle_data(true);
    /// ```
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify whether or not to be verbose
    /// ```
    /// # use automl::classification::Settings;
    /// let settings = Settings::default().verbose(true);
    /// ```
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::Algorithm;
    /// let settings = Settings::default().skip(Algorithm::RandomForest);
    /// ```
    pub fn skip(mut self, skip: Algorithm) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Adds a specific sorting function to the settings
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::Metric;
    /// let settings = Settings::default().sorted_by(Metric::Accuracy);
    /// ```
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for random_forest
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::RandomForestClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_settings(RandomForestClassifierParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_settings(
        mut self,
        settings: RandomForestClassifierParameters,
    ) -> Self {
        self.random_forest_settings = settings;
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::LogisticRegressionParameters;
    /// let settings = Settings::default()
    ///     .with_logistic_settings(LogisticRegressionParameters::default());
    /// ```
    pub fn with_logistic_settings(mut self, settings: LogisticRegressionParameters) -> Self {
        self.logistic_settings = settings;
        self
    }

    /// Specify settings for support vector classifier
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::{SVCParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svc_settings(SVCParameters::default()
    ///         .with_epoch(10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svc_settings(mut self, settings: SVCParameters) -> Self {
        self.svc_settings = settings;
        self
    }

    /// Specify settings for decision tree classifier
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::DecisionTreeClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_settings(DecisionTreeClassifierParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_settings = settings;
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::{KNNClassifierParameters,
    ///     KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_settings(KNNClassifierParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_settings(mut self, settings: KNNClassifierParameters) -> Self {
        self.knn_settings = settings;
        self
    }

    /// Specify settings for Gaussian Naive Bayes
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::GaussianNBParameters;
    /// let settings = Settings::default()
    ///     .with_gaussian_nb_settings(GaussianNBParameters::default()
    ///         .with_priors(vec![1.0, 1.0])
    ///     );
    /// ```
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters<f32>) -> Self {
        self.gaussian_nb_settings = settings;
        self
    }

    /// Specify settings for Categorical Naive Bayes
    /// ```
    /// # use automl::classification::Settings;
    /// use automl::classification::settings::CategoricalNBParameters;
    /// let settings = Settings::default()
    ///     .with_categorical_nb_settings(CategoricalNBParameters::default()
    ///         .with_alpha(1.0)
    ///     );
    /// ```
    pub fn with_categorical_nb_settings(mut self, settings: CategoricalNBParameters<f32>) -> Self {
        self.categorical_nb_settings = settings;
        self
    }
}

impl Display for Settings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Prep new table
        let mut table = Table::new();

        // Get list of algorithms to skip
        let mut skiplist = String::new();
        if self.skiplist.len() == 0 {
            skiplist.push_str("None");
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
            .add_row(vec!["    Verbose", &*format!("{}", self.verbose)])
            .add_row(vec!["    Sorting Metric", &*format!("{}", self.sort_by)])
            .add_row(vec!["    Shuffle Data", &*format!("{}", self.shuffle)])
            .add_row(vec![
                "    Number of CV Folds",
                &*format!("{}", self.number_of_folds),
            ])
            .add_row(vec![
                "    Skipped Algorithms",
                &*format!("{}", &skiplist[0..skiplist.len() - 1]),
            ]);
        write!(f, "{}\n", table)
    }
}

impl epi::App for Classifier {
    fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame<'_>) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let value_to_predict = vec![self.current_x.to_vec(); 1];

            ui.heading(format!("{}", self.comparison[0].name));
            ui.label(format!(
                "Prediction: y = {:?}",
                self.predict(&DenseMatrix::from_2d_vec(&value_to_predict))[0]
            ));
            ui.separator();

            for i in 0..self.current_x.len() {
                let maxx = self
                    .x
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::max);

                let minn = self
                    .x
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::min);
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

impl Classifier {
    pub fn run_demo(self) {
        let native_options = eframe::NativeOptions::default();
        eframe::run_native(Box::new(self), native_options);
    }
}
