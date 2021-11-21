//! Auto-ML for regression models

use crate::utils::{print_knn_search_algorithm, print_knn_weight_function, print_option, Status};
pub use crate::utils::{Distance, Kernel};
use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};
use humantime::format_duration;
use polars::prelude::{CsvReader, DataFrame, Float32Type, SerReader};
use smartcore::math::distance::{
    hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan, minkowski::Minkowski,
    Distances,
};
use smartcore::svm::{LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel};
pub use smartcore::{
    algorithm::neighbour::KNNAlgorithmName,
    ensemble::random_forest_regressor::RandomForestRegressorParameters,
    linear::{
        elastic_net::ElasticNetParameters,
        lasso::LassoParameters,
        linear_regression::{LinearRegressionParameters, LinearRegressionSolverName},
        ridge_regression::{RidgeRegressionParameters, RidgeRegressionSolverName},
    },
    neighbors::KNNWeightFunction,
    tree::decision_tree_regressor::DecisionTreeRegressorParameters,
};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::naive::dense_matrix::DenseMatrix,
    linear::{
        elastic_net::ElasticNet, lasso::Lasso, linear_regression::LinearRegression,
        ridge_regression::RidgeRegression,
    },
    math::distance::euclidian::Euclidian,
    metrics::{mean_absolute_error, mean_squared_error, r2},
    model_selection::{cross_validate, CrossValidationResult, KFold},
    neighbors::knn_regressor::{
        KNNRegressor, KNNRegressorParameters as SmartcoreKNNRegressorParameters,
    },
    svm::{
        svr::{SVRParameters as SmartcoreSVRParameters, SVR},
        Kernels,
    },
    tree::decision_tree_regressor::DecisionTreeRegressor,
};
use std::time::{Duration, Instant};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
};

/// An enum for sorting
#[non_exhaustive]
#[derive(PartialEq)]
pub enum Metric {
    /// Sort by R^2
    RSquared,
    /// Sort by MAE
    MeanAbsoluteError,
    /// Sort by MSE
    MeanSquaredError,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::RSquared => write!(f, "R^2"),
            Metric::MeanAbsoluteError => write!(f, "MAE"),
            Metric::MeanSquaredError => write!(f, "MSE"),
        }
    }
}

/// An enum containing possible regression algorithms
#[derive(PartialEq)]
pub enum Algorithm {
    /// Decision tree regressor
    DecisionTree,
    /// KNN Regressor
    KNN,
    /// Random forest regressor
    RandomForest,
    /// Linear regressor
    Linear,
    /// Ridge regressor
    Ridge,
    /// Lasso regressor
    Lasso,
    /// Elastic net regressor
    ElasticNet,
    /// Support vector regressor
    SVR,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::DecisionTree => write!(f, "Decision Tree Regressor"),
            Algorithm::KNN => write!(f, "KNN Regressor"),
            Algorithm::RandomForest => write!(f, "Random Forest Regressor"),
            Algorithm::Linear => write!(f, "Linear Regressor"),
            Algorithm::Ridge => write!(f, "Ridge Regressor"),
            Algorithm::Lasso => write!(f, "LASSO Regressor"),
            Algorithm::ElasticNet => write!(f, "Elastic Net Regressor"),
            Algorithm::SVR => write!(f, "Support Vector Regressor"),
        }
    }
}

/// This is the output from a model comparison operation
pub struct Regressor {
    settings: Settings,
    x: DenseMatrix<f32>,
    y: Vec<f32>,
    comparison: Vec<Model>,
    final_model: Vec<u8>,
    status: Status,
}

impl Regressor {
    /// Create a new regressor based on settings
    pub fn new(x: DenseMatrix<f32>, y: Vec<f32>, settings: Settings) -> Self {
        Self {
            settings,
            x,
            y,
            comparison: vec![],
            final_model: vec![],
            status: Status::DataLoaded,
        }
    }

    /// Add settings to the regressor
    pub fn with_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    /// Runs a model comparison and trains a final model. [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(&mut self) {
        self.compare_models();
        self.train_final_model();
    }

    /// Add data to regressor object
    pub fn with_data(&mut self, x: DenseMatrix<f32>, y: Vec<f32>) {
        self.x = x;
        self.y = y;
        self.status = Status::DataLoaded;
    }

    /// Add data from a csv
    pub fn with_data_from_csv(&mut self, filepath: &str, target: usize, header: bool) {
        let df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

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

        // Set status
        self.status = Status::DataLoaded;
    }

    /// Add a dataset to regressor object
    pub fn with_dataset(&mut self, dataset: Dataset<f32, f32>) {
        self.x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        self.y = dataset.target;
        // Set status
        self.status = Status::DataLoaded;
    }

    /// This function compares all of the regression models available in the package.
    pub fn compare_models(&mut self) {
        if self.status == Status::DataLoaded {
            if !self.settings.skiplist.contains(&Algorithm::Linear) {
                let start = Instant::now();
                let cv = cross_validate(
                    LinearRegression::fit,
                    &self.x,
                    &self.y,
                    self.settings.linear_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
                )
                .unwrap();
                let end = Instant::now();
                self.add_model(Algorithm::Linear, cv, end.duration_since(start));
            }

            if !self.settings.skiplist.contains(&Algorithm::SVR) {
                let start = Instant::now();
                let cv = match self.settings.svr_settings.kernel {
                    Kernel::Linear => cross_validate(
                        SVR::fit,
                        &self.x,
                        &self.y,
                        SmartcoreSVRParameters::default()
                            .with_tol(self.settings.svr_settings.tol)
                            .with_c(self.settings.svr_settings.c)
                            .with_eps(self.settings.svr_settings.c)
                            .with_kernel(Kernels::linear()),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Kernel::Polynomial(degree, gamma, coef) => cross_validate(
                        SVR::fit,
                        &self.x,
                        &self.y,
                        SmartcoreSVRParameters::default()
                            .with_tol(self.settings.svr_settings.tol)
                            .with_c(self.settings.svr_settings.c)
                            .with_eps(self.settings.svr_settings.c)
                            .with_kernel(Kernels::polynomial(degree, gamma, coef)),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Kernel::RBF(gamma) => cross_validate(
                        SVR::fit,
                        &self.x,
                        &self.y,
                        SmartcoreSVRParameters::default()
                            .with_tol(self.settings.svr_settings.tol)
                            .with_c(self.settings.svr_settings.c)
                            .with_eps(self.settings.svr_settings.c)
                            .with_kernel(Kernels::rbf(gamma)),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Kernel::Sigmoid(gamma, coef) => cross_validate(
                        SVR::fit,
                        &self.x,
                        &self.y,
                        SmartcoreSVRParameters::default()
                            .with_tol(self.settings.svr_settings.tol)
                            .with_c(self.settings.svr_settings.c)
                            .with_eps(self.settings.svr_settings.c)
                            .with_kernel(Kernels::sigmoid(gamma, coef)),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
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
                    self.settings.lasso_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
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
                    self.settings.ridge_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
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
                    self.settings.elastic_net_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
                )
                .unwrap();
                let end = Instant::now();
                let d = end.duration_since(start);
                self.add_model(Algorithm::ElasticNet, cv, d);
            }

            if !self.settings.skiplist.contains(&Algorithm::DecisionTree) {
                let start = Instant::now();
                let cv = cross_validate(
                    DecisionTreeRegressor::fit,
                    &self.x,
                    &self.y,
                    self.settings.decision_tree_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
                )
                .unwrap();
                let end = Instant::now();
                let d = end.duration_since(start);
                self.add_model(Algorithm::DecisionTree, cv, d);
            }

            if !self.settings.skiplist.contains(&Algorithm::RandomForest) {
                let start = Instant::now();
                let cv = cross_validate(
                    RandomForestRegressor::fit,
                    &self.x,
                    &self.y,
                    self.settings.random_forest_settings.clone(),
                    self.get_kfolds(),
                    match self.settings.sort_by {
                        Metric::RSquared => r2,
                        Metric::MeanAbsoluteError => mean_absolute_error,
                        Metric::MeanSquaredError => mean_squared_error,
                    },
                )
                .unwrap();
                let end = Instant::now();
                let d = end.duration_since(start);
                self.add_model(Algorithm::RandomForest, cv, d);
            }

            if !self.settings.skiplist.contains(&Algorithm::KNN) {
                let start = Instant::now();
                let cv = match self.settings.knn_settings.distance {
                    Distance::Euclidean => cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNRegressorParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_distance(Distances::euclidian()),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Distance::Manhattan => cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNRegressorParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_distance(Distances::manhattan()),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Distance::Minkowski(p) => cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNRegressorParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_distance(Distances::minkowski(p)),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Distance::Mahalanobis => cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNRegressorParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_distance(Distances::mahalanobis(&self.x)),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                    Distance::Hamming => cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        SmartcoreKNNRegressorParameters::default()
                            .with_k(self.settings.knn_settings.k)
                            .with_algorithm(self.settings.knn_settings.algorithm.clone())
                            .with_weight(self.settings.knn_settings.weight.clone())
                            .with_distance(Distances::hamming()),
                        self.get_kfolds(),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                };
                let end = Instant::now();
                let d = end.duration_since(start);

                self.add_model(Algorithm::KNN, cv, d);
            }
            self.status = Status::ModelsCompared;
        } else {
            panic!("You must load data before trying to compare models.")
        }
    }

    /// Trains the best model found during comparison
    pub fn train_final_model(&mut self) {
        match self.comparison[0].name {
            Algorithm::Linear => {
                self.final_model = bincode::serialize(
                    &LinearRegression::fit(&self.x, &self.y, self.settings.linear_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            Algorithm::Lasso => {
                self.final_model = bincode::serialize(
                    &Lasso::fit(&self.x, &self.y, self.settings.lasso_settings.clone()).unwrap(),
                )
                .unwrap()
            }
            Algorithm::Ridge => {
                self.final_model = bincode::serialize(
                    &RidgeRegression::fit(&self.x, &self.y, self.settings.ridge_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            Algorithm::ElasticNet => {
                self.final_model = bincode::serialize(
                    &ElasticNet::fit(&self.x, &self.y, self.settings.elastic_net_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            Algorithm::RandomForest => {
                self.final_model = bincode::serialize(
                    &RandomForestRegressor::fit(
                        &self.x,
                        &self.y,
                        self.settings.random_forest_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::KNN => match self.settings.knn_settings.distance {
                Distance::Euclidean => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_distance(Distances::euclidian());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Manhattan => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_distance(Distances::manhattan());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Minkowski(p) => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_distance(Distances::minkowski(p));

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Mahalanobis => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_distance(Distances::mahalanobis(&self.x));

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
                Distance::Hamming => {
                    let params = SmartcoreKNNRegressorParameters::default()
                        .with_k(self.settings.knn_settings.k)
                        .with_algorithm(self.settings.knn_settings.algorithm.clone())
                        .with_weight(self.settings.knn_settings.weight.clone())
                        .with_distance(Distances::hamming());

                    self.final_model =
                        bincode::serialize(&KNNRegressor::fit(&self.x, &self.y, params).unwrap())
                            .unwrap()
                }
            },
            Algorithm::SVR => match self.settings.svr_settings.kernel {
                Kernel::Linear => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.tol)
                        .with_c(self.settings.svr_settings.c)
                        .with_eps(self.settings.svr_settings.c)
                        .with_kernel(Kernels::linear());
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Polynomial(degree, gamma, coef) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.tol)
                        .with_c(self.settings.svr_settings.c)
                        .with_eps(self.settings.svr_settings.c)
                        .with_kernel(Kernels::polynomial(degree, gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::RBF(gamma) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.tol)
                        .with_c(self.settings.svr_settings.c)
                        .with_eps(self.settings.svr_settings.c)
                        .with_kernel(Kernels::rbf(gamma));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
                Kernel::Sigmoid(gamma, coef) => {
                    let params = SmartcoreSVRParameters::default()
                        .with_tol(self.settings.svr_settings.tol)
                        .with_c(self.settings.svr_settings.c)
                        .with_eps(self.settings.svr_settings.c)
                        .with_kernel(Kernels::sigmoid(gamma, coef));
                    self.final_model =
                        bincode::serialize(&SVR::fit(&self.x, &self.y, params).unwrap()).unwrap()
                }
            },
            Algorithm::DecisionTree => {
                self.final_model = bincode::serialize(
                    &DecisionTreeRegressor::fit(
                        &self.x,
                        &self.y,
                        self.settings.decision_tree_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
        }

        self.status = Status::FinalModelTrained;
    }

    /// Predict values using the best model
    pub fn predict(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.comparison[0].name {
            Algorithm::Linear => {
                let model: LinearRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
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
            Algorithm::RandomForest => {
                let model: RandomForestRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::KNN => match self.settings.knn_settings.distance {
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
            Algorithm::SVR => match self.settings.svr_settings.kernel {
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
            Algorithm::DecisionTree => {
                let model: DecisionTreeRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}

/// Private regressor functions go here
impl Regressor {
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
            a.score
                .mean_test_score()
                .partial_cmp(&b.score.mean_test_score())
                .unwrap_or(Equal)
        });
        if self.settings.sort_by == Metric::RSquared {
            self.comparison.reverse();
        }
    }
}

impl Display for Regressor {
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

impl Default for Regressor {
    fn default() -> Self {
        Self {
            settings: Default::default(),
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: vec![],
            final_model: vec![],
            status: Status::Starting,
        }
    }
}

/// This contains the results of a single model
struct Model {
    score: CrossValidationResult<f32>,
    name: Algorithm,
    duration: Duration,
}

/// The settings artifact for all regressions
pub struct Settings {
    sort_by: Metric,
    skiplist: Vec<Algorithm>,
    number_of_folds: usize,
    shuffle: bool,
    verbose: bool,
    linear_settings: LinearRegressionParameters,
    svr_settings: SVRParameters,
    lasso_settings: LassoParameters<f32>,
    ridge_settings: RidgeRegressionParameters<f32>,
    elastic_net_settings: ElasticNetParameters<f32>,
    decision_tree_settings: DecisionTreeRegressorParameters,
    random_forest_settings: RandomForestRegressorParameters,
    knn_settings: KNNRegressorParameters,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            skiplist: vec![],
            number_of_folds: 10,
            shuffle: true,
            verbose: true,
            linear_settings: LinearRegressionParameters::default(),
            svr_settings: SVRParameters::default(),
            lasso_settings: LassoParameters::default(),
            ridge_settings: RidgeRegressionParameters::default(),
            elastic_net_settings: ElasticNetParameters::default(),
            decision_tree_settings: DecisionTreeRegressorParameters::default(),
            random_forest_settings: RandomForestRegressorParameters::default(),
            knn_settings: KNNRegressorParameters::default(),
        }
    }
}

impl Settings {
    /// Specify number of folds for cross-validation
    /// ```
    /// # use automl::regression::Settings;
    /// let settings = Settings::default().with_number_of_folds(3);
    /// ```
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    /// ```
    /// # use automl::regression::Settings;
    /// let settings = Settings::default().shuffle_data(true);
    /// ```
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify whether or not to be verbose
    /// ```
    /// # use automl::regression::Settings;
    /// let settings = Settings::default().verbose(true);
    /// ```
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::Algorithm;
    /// let settings = Settings::default().skip(Algorithm::RandomForest);
    /// ```
    pub fn skip(mut self, skip: Algorithm) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Adds a specific sorting function to the settings
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::Metric;
    /// let settings = Settings::default().sorted_by(Metric::RSquared);
    /// ```
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for linear regression
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::{LinearRegressionParameters, LinearRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_linear_settings(LinearRegressionParameters::default()
    ///         .with_solver(LinearRegressionSolverName::QR)
    ///     );
    /// ```
    pub fn with_linear_settings(mut self, settings: LinearRegressionParameters) -> Self {
        self.linear_settings = settings;
        self
    }

    /// Specify settings for lasso regression
    /// ```
    /// # use automl::regression::Algorithm::Lasso;
    /// use automl::regression::Settings;
    /// use automl::regression::LassoParameters;
    /// let settings = Settings::default()
    ///     .with_lasso_settings(LassoParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_tol(1e-10)
    ///         .with_normalize(true)
    ///         .with_max_iter(10_000)
    ///     );
    /// ```
    pub fn with_lasso_settings(mut self, settings: LassoParameters<f32>) -> Self {
        self.lasso_settings = settings;
        self
    }

    /// Specify settings for ridge regression
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::{RidgeRegressionParameters, RidgeRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_ridge_settings(RidgeRegressionParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_normalize(true)
    ///         .with_solver(RidgeRegressionSolverName::Cholesky)
    ///     );
    /// ```
    pub fn with_ridge_settings(mut self, settings: RidgeRegressionParameters<f32>) -> Self {
        self.ridge_settings = settings;
        self
    }

    /// Specify settings for elastic net
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::ElasticNetParameters;
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
        self.elastic_net_settings = settings;
        self
    }

    /// Specify settings for KNN regressor
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::{KNNRegressorParameters, KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_settings(KNNRegressorParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_settings(mut self, settings: KNNRegressorParameters) -> Self {
        self.knn_settings = settings;
        self
    }

    /// Specify settings for support vector regressor
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::{SVRParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svr_settings(SVRParameters::default()
    ///         .with_eps(1e-10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svr_settings(mut self, settings: SVRParameters) -> Self {
        self.svr_settings = settings;
        self
    }

    /// Specify settings for random forest
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::RandomForestRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_settings(RandomForestRegressorParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_settings(
        mut self,
        settings: RandomForestRegressorParameters,
    ) -> Self {
        self.random_forest_settings = settings;
        self
    }

    /// Specify settings for decision tree
    /// ```
    /// # use automl::regression::Settings;
    /// use automl::regression::DecisionTreeRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_settings(DecisionTreeRegressorParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_settings(
        mut self,
        settings: DecisionTreeRegressorParameters,
    ) -> Self {
        self.decision_tree_settings = settings;
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
        if !self.skiplist.contains(&Algorithm::Linear) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Linear).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Solver",
                    match self.linear_settings.solver {
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
                    match self.ridge_settings.solver {
                        RidgeRegressionSolverName::Cholesky => "Cholesky",
                        RidgeRegressionSolverName::SVD => "SVD",
                    },
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.ridge_settings.alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.ridge_settings.normalize),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::Lasso) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Lasso).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.lasso_settings.alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.lasso_settings.normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.lasso_settings.max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.lasso_settings.tol),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::ElasticNet) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::ElasticNet).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.elastic_net_settings.alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.elastic_net_settings.normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.elastic_net_settings.max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.elastic_net_settings.tol),
                ])
                .add_row(vec![
                    "    L1 Ratio",
                    &*format!("{}", self.elastic_net_settings.l1_ratio),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::DecisionTree) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::DecisionTree).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(self.decision_tree_settings.max_depth),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!("{}", self.decision_tree_settings.min_samples_leaf),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!("{}", self.decision_tree_settings.min_samples_split),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::RandomForest) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::RandomForest).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(self.random_forest_settings.max_depth),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!("{}", self.random_forest_settings.min_samples_leaf),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!("{}", self.random_forest_settings.min_samples_split),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!("{}", self.random_forest_settings.n_trees),
                ])
                .add_row(vec![
                    "    Number of split candidates",
                    &*print_option(self.random_forest_settings.m),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::KNN) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::KNN).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Number of neighbors",
                    &*format!("{}", self.knn_settings.k),
                ])
                .add_row(vec![
                    "    Search algorithm",
                    &*format!(
                        "{}",
                        print_knn_search_algorithm(&self.knn_settings.algorithm)
                    ),
                ])
                .add_row(vec![
                    "    Weighting function",
                    &*format!("{}", print_knn_weight_function(&self.knn_settings.weight)),
                ])
                .add_row(vec![
                    "    Distance function",
                    &*format!("{}", &self.knn_settings.distance),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::SVR) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::SVR).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Regularization parameter",
                    &*format!("{}", self.svr_settings.c),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.svr_settings.tol),
                ])
                .add_row(vec!["    Epsilon", &*format!("{}", self.svr_settings.eps)])
                .add_row(vec![
                    "    Kernel",
                    &*format!("{}", self.svr_settings.kernel),
                ]);
        }

        write!(f, "{}\n", table)
    }
}

/// Parameters for KNN Regression
pub struct KNNRegressorParameters {
    k: usize,
    weight: KNNWeightFunction,
    algorithm: KNNAlgorithmName,
    distance: Distance,
}

impl KNNRegressorParameters {
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

impl Default for KNNRegressorParameters {
    fn default() -> Self {
        Self {
            k: 3,
            weight: KNNWeightFunction::Uniform,
            algorithm: KNNAlgorithmName::CoverTree,
            distance: Distance::Euclidean,
        }
    }
}

/// A struct for
pub struct SVRParameters {
    eps: f32,
    c: f32,
    tol: f32,
    kernel: Kernel,
}

impl SVRParameters {
    /// Define the value of epsilon to use in the epsilon-SVR model.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
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

impl Default for SVRParameters {
    fn default() -> Self {
        Self {
            eps: 0.1,
            c: 1.0,
            tol: 1e-3,
            kernel: Kernel::Linear,
        }
    }
}
