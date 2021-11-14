//! Auto-ML for regression models

use super::traits::ValidRegressor;
use crate::regression::Algorithm::Ridge;
use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use polars::prelude::*;
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::naive::dense_matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet, ElasticNetParameters},
        lasso::{Lasso, LassoParameters},
        linear_regression::{LinearRegression, LinearRegressionParameters},
        ridge_regression::{RidgeRegression, RidgeRegressionParameters},
    },
    math::distance::euclidian::Euclidian,
    metrics::{
        mean_absolute_error::MeanAbsoluteError, mean_squared_error::MeanSquareError, r2::R2,
    },
    model_selection::train_test_split,
    neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters},
    svm::{
        svr::{SVRParameters, SVR},
        LinearKernel,
    },
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};

/// An enum for sorting
#[non_exhaustive]
pub enum Metric {
    /// Sort by R^2
    RSquared,
    /// Sort by MAE
    MeanAbsoluteError,
    /// Sort by MSE
    MeanSquaredError,
}

/// An enum containing regression algorithms
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
    SupportVector,
}

/// This is the output from a model comparison operation
pub struct Regressor {
    settings: Settings,
    x: DenseMatrix<f32>,
    y: Vec<f32>,
    comparison: Vec<Model>,
    final_model: Vec<u8>,
}

impl Regressor {
    /// [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(settings: Settings, x: DenseMatrix<f32>, y: Vec<f32>) -> Self {
        let mut regressor = Self::new(settings);
        regressor.with_data(x, y);
        regressor.compare_models();
        regressor.train_final_model();
        regressor
    }

    /// Predict values using the best model
    pub fn predict(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.comparison[0].name.as_str() {
            "Linear Regressor" => {
                let model: LinearRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "LASSO Regressor" => {
                let model: Lasso<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "Ridge Regressor" => {
                let model: RidgeRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "Elastic Net Regressor" => {
                let model: ElasticNet<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "Random Forest Regressor" => {
                let model: RandomForestRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "KNN Regressor" => {
                let model: KNNRegressor<f32, Euclidian> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "Support Vector Regressor" => {
                let model: SVR<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            "Decision Tree Regressor" => {
                let model: DecisionTreeRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            &_ => panic!("Unable to predict"),
        }
    }

    /// Uses the best model to make a prediction
    pub fn train_final_model(&mut self) {
        match self.comparison[0].name.as_str() {
            "Linear Regressor" => {
                self.final_model = bincode::serialize(
                    &LinearRegression::fit(&self.x, &self.y, self.settings.linear_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            "LASSO Regressor" => {
                self.final_model = bincode::serialize(
                    &Lasso::fit(&self.x, &self.y, self.settings.lasso_settings.clone()).unwrap(),
                )
                .unwrap()
            }
            "Ridge Regressor" => {
                self.final_model = bincode::serialize(
                    &RidgeRegression::fit(&self.x, &self.y, self.settings.ridge_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            "Elastic Net Regressor" => {
                self.final_model = bincode::serialize(
                    &ElasticNet::fit(&self.x, &self.y, self.settings.elastic_net_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            "Random Forest Regressor" => {
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
            "KNN Regressor" => {
                self.final_model = bincode::serialize(
                    &KNNRegressor::fit(&self.x, &self.y, self.settings.knn_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            "Support Vector Regressor" => {
                self.final_model = bincode::serialize(
                    &SVR::fit(&self.x, &self.y, self.settings.svr_settings.clone()).unwrap(),
                )
                .unwrap()
            }
            "Decision Tree Regressor" => {
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
            &_ => panic!("Unable to predict"),
        }
    }

    /// Returns a serialized version of the best model
    pub fn get_best_model(&self) -> Vec<u8> {
        self.final_model.clone()
    }

    fn add_model(&mut self, name: String, y_test: &Vec<f32>, y_pred: &Vec<f32>) {
        self.comparison.push(Model {
            r_squared: R2 {}.get_score(y_test, y_pred),
            mean_absolute_error: MeanAbsoluteError {}.get_score(y_test, y_pred),
            mean_squared_error: MeanSquareError {}.get_score(y_test, y_pred),
            name,
        });
        self.sort()
    }

    fn sort(&mut self) {
        match self.settings.sort_by {
            Metric::RSquared => {
                self.comparison
                    .sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap_or(Equal));
            }
            Metric::MeanSquaredError => {
                self.comparison.sort_by(|a, b| {
                    a.mean_squared_error
                        .partial_cmp(&b.mean_squared_error)
                        .unwrap_or(Equal)
                });
            }
            Metric::MeanAbsoluteError => {
                self.comparison.sort_by(|a, b| {
                    a.mean_absolute_error
                        .partial_cmp(&b.mean_absolute_error)
                        .unwrap_or(Equal)
                });
            }
        }
    }

    /// Create a new regressor based on settings
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: Vec::new(),
            final_model: vec![],
        }
    }

    /// Add data to regressor object
    pub fn with_data(&mut self, x: DenseMatrix<f32>, y: Vec<f32>) {
        self.x = x;
        self.y = y;
    }

    /// Add data from a csv
    pub fn with_data_from_csv(&mut self, filepath: &str, target: usize, header: bool) {
        let mut df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

        // Get target variables
        let target_column_name = df.get_column_names()[target];
        let series = df.column(target_column_name).unwrap();
        match series.dtype() {
            DataType::Boolean => series
                .bool()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(if v { 1.0 as f32 } else { 0.0 as f32 })),
            DataType::UInt8 => series
                .u8()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::UInt16 => series
                .u64()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::UInt32 => series
                .u32()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::UInt64 => series
                .u64()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Int8 => series
                .i8()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Int16 => series
                .i16()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Int32 => series
                .i32()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Int64 => series
                .i64()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Float32 => series
                .f32()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Float64 => series
                .f64()
                .unwrap()
                .into_no_null_iter()
                .for_each(|v| self.y.push(v as f32)),
            DataType::Utf8 => {
                panic!("Text data encountered")
            }
            DataType::Date => {
                panic!("No idea how to handle dates or times yet")
            }
            DataType::Datetime => {
                panic!("No idea how to handle dates or times yet")
            }
            DataType::Time => {
                panic!("No idea how to handle dates or times yet")
            }
            DataType::List(_) => {
                panic!("Super weird failure")
            }
            DataType::Null => panic!("Null data encountered"),

            DataType::Categorical => panic!("Categorical data is bad mmk"),
            _ => panic!("Object encountered?"),
        }

        // Get the rest of the data
        let features = df.drop(target_column_name).unwrap();
        let (height, width) = features.shape();
        let ndarray = features.to_ndarray::<Float32Type>().unwrap();
        self.x = DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap());
    }

    /// Add a dataset to regressor object
    pub fn with_dataset(&mut self, dataset: Dataset<f32, f32>) {
        self.x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        self.y = dataset.target;
    }

    /// This function compares all of the regression models available in the package.
    pub fn compare_models(&mut self) {
        let (x_test, x_train, y_test, y_train) = train_test_split(
            &self.x,
            &self.y,
            self.settings.testing_fraction,
            self.settings.shuffle,
        );

        if !self.settings.skiplist.contains(&Algorithm::Linear) {
            let model =
                LinearRegression::fit(&x_train, &y_train, self.settings.linear_settings.clone())
                    .unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::SupportVector) {
            let model = SVR::fit(&x_train, &y_train, self.settings.svr_settings.clone()).unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::Lasso) {
            let model =
                Lasso::fit(&x_train, &y_train, self.settings.lasso_settings.clone()).unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::Ridge) {
            let model =
                RidgeRegression::fit(&x_train, &y_train, self.settings.ridge_settings.clone())
                    .unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::ElasticNet) {
            let model = ElasticNet::fit(
                &x_train,
                &y_train,
                self.settings.elastic_net_settings.clone(),
            )
            .unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::DecisionTree) {
            let model = DecisionTreeRegressor::fit(
                &x_train,
                &y_train,
                self.settings.decision_tree_settings.clone(),
            )
            .unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::RandomForest) {
            let model = RandomForestRegressor::fit(
                &x_train,
                &y_train,
                self.settings.random_forest_settings.clone(),
            )
            .unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }

        if !self.settings.skiplist.contains(&Algorithm::KNN) {
            let model =
                KNNRegressor::fit(&x_train, &y_train, self.settings.knn_settings.clone()).unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(model.name(), &y_test, &y_pred);
        }
    }
}

impl Display for Regressor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "R^2", "MSE", "MAE"]);
        for model in &self.comparison {
            table.add_row(vec![
                format!("{}", &model.name),
                format!("{:.3}", &model.r_squared),
                format!("{:.3e}", &model.mean_squared_error),
                format!("{:.3e}", &model.mean_absolute_error),
            ]);
        }
        write!(f, "{}\n", table)
    }
}

/// This contains the results of a single model, including the model itself
struct Model {
    r_squared: f32,
    mean_absolute_error: f32,
    mean_squared_error: f32,
    name: String,
}

/// The settings artifact for all regressions
pub struct Settings {
    sort_by: Metric,
    skiplist: Vec<Algorithm>,
    testing_fraction: f32,
    shuffle: bool,
    linear_settings: LinearRegressionParameters,
    svr_settings: SVRParameters<f32, DenseMatrix<f32>, LinearKernel>,
    lasso_settings: LassoParameters<f32>,
    ridge_settings: RidgeRegressionParameters<f32>,
    elastic_net_settings: ElasticNetParameters<f32>,
    decision_tree_settings: DecisionTreeRegressorParameters,
    random_forest_settings: RandomForestRegressorParameters,
    knn_settings: KNNRegressorParameters<f32, Euclidian>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            skiplist: vec![],
            testing_fraction: 0.3,
            shuffle: true,
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
    /// Specify algorithms that shouldn't be included in comparison
    pub fn skip(mut self, skip: Vec<Algorithm>) -> Self {
        self.skiplist = skip;
        self
    }

    /// Adds a specific sorting function to the settings
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for linear regression
    pub fn with_linear_settings(mut self, settings: LinearRegressionParameters) -> Self {
        self.linear_settings = settings;
        self
    }

    /// Specify settings for lasso regression
    pub fn with_lasso_settings(mut self, settings: LassoParameters<f32>) -> Self {
        self.lasso_settings = settings;
        self
    }

    /// Specify settings for ridge regression
    pub fn with_ridge_settings(mut self, settings: RidgeRegressionParameters<f32>) -> Self {
        self.ridge_settings = settings;
        self
    }

    /// Specify settings for elastic net
    pub fn with_elastic_net_settings(mut self, settings: ElasticNetParameters<f32>) -> Self {
        self.elastic_net_settings = settings;
        self
    }

    /// Specify settings for KNN regressor
    pub fn with_knn_settings(mut self, settings: KNNRegressorParameters<f32, Euclidian>) -> Self {
        self.knn_settings = settings;
        self
    }

    /// Specify settings for support vector regressor
    pub fn with_svr_settings(
        mut self,
        settings: SVRParameters<f32, DenseMatrix<f32>, LinearKernel>,
    ) -> Self {
        self.svr_settings = settings;
        self
    }

    /// Specify settings for random forest
    pub fn with_random_forest_settings(
        mut self,
        settings: RandomForestRegressorParameters,
    ) -> Self {
        self.random_forest_settings = settings;
        self
    }

    /// Specify settings for decision tree
    pub fn with_decision_tree_settings(
        mut self,
        settings: DecisionTreeRegressorParameters,
    ) -> Self {
        self.decision_tree_settings = settings;
        self
    }
}
