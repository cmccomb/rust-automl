//! Auto-ML for regression models

use super::utils::Status;
use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};
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
    metrics::{mean_absolute_error, mean_squared_error, r2},
    model_selection::{cross_validate, CrossValidationResult, KFold},
    neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters},
    svm::{
        svr::{SVRParameters, SVR},
        LinearKernel,
    },
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};
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
    /// [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(settings: Settings, x: DenseMatrix<f32>, y: Vec<f32>) -> Self {
        let mut regressor = Self::new(settings);
        regressor.with_data(x, y);
        regressor.compare_models();
        regressor.train_final_model();
        regressor
    }

    /// Create a new regressor based on settings
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: vec![],
            final_model: vec![],
            status: Status::Starting,
        }
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
            Algorithm::KNN => {
                let model: KNNRegressor<f32, Euclidian> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::SVR => {
                let model: SVR<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::DecisionTree => {
                let model: DecisionTreeRegressor<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
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
            Algorithm::KNN => {
                self.final_model = bincode::serialize(
                    &KNNRegressor::fit(&self.x, &self.y, self.settings.knn_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            Algorithm::SVR => {
                self.final_model = bincode::serialize(
                    &SVR::fit(&self.x, &self.y, self.settings.svr_settings.clone()).unwrap(),
                )
                .unwrap()
            }
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

    /// Returns a serialized version of the best model
    pub fn get_best_model(&self) -> Vec<u8> {
        self.final_model.clone()
    }

    fn add_model(&mut self, name: Algorithm, score: CrossValidationResult<f32>) {
        self.comparison.push(Model { score, name });
        self.sort()
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
            _ => panic!("Idk what happened"),
        }

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
                self.add_model(
                    Algorithm::Linear,
                    cross_validate(
                        LinearRegression::fit,
                        &self.x,
                        &self.y,
                        self.settings.linear_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::SVR) {
                self.add_model(
                    Algorithm::SVR,
                    cross_validate(
                        SVR::fit,
                        &self.x,
                        &self.y,
                        self.settings.svr_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::Lasso) {
                self.add_model(
                    Algorithm::Lasso,
                    cross_validate(
                        Lasso::fit,
                        &self.x,
                        &self.y,
                        self.settings.lasso_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::Ridge) {
                self.add_model(
                    Algorithm::Ridge,
                    cross_validate(
                        RidgeRegression::fit,
                        &self.x,
                        &self.y,
                        self.settings.ridge_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::ElasticNet) {
                self.add_model(
                    Algorithm::ElasticNet,
                    cross_validate(
                        ElasticNet::fit,
                        &self.x,
                        &self.y,
                        self.settings.elastic_net_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::DecisionTree) {
                self.add_model(
                    Algorithm::DecisionTree,
                    cross_validate(
                        DecisionTreeRegressor::fit,
                        &self.x,
                        &self.y,
                        self.settings.decision_tree_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::RandomForest) {
                self.add_model(
                    Algorithm::RandomForest,
                    cross_validate(
                        RandomForestRegressor::fit,
                        &self.x,
                        &self.y,
                        self.settings.random_forest_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::KNN) {
                self.add_model(
                    Algorithm::KNN,
                    cross_validate(
                        KNNRegressor::fit,
                        &self.x,
                        &self.y,
                        self.settings.knn_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::RSquared => r2,
                            Metric::MeanAbsoluteError => mean_absolute_error,
                            Metric::MeanSquaredError => mean_squared_error,
                        },
                    )
                    .unwrap(),
                );
            }
            self.status = Status::ModelsCompared;
        } else {
            panic!("You must load data before trying to compare models.")
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
            Cell::new(format!("Training {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
        ]);
        for model in &self.comparison {
            let mut row_vec = vec![];
            row_vec.push(format!("{}", &model.name));
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
}

/// The settings artifact for all regressions
pub struct Settings {
    sort_by: Metric,
    skiplist: Vec<Algorithm>,
    number_of_folds: usize,
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
            number_of_folds: 10,
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
    /// Specify number of folds for cross-validation
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    pub fn skip_algorithms(mut self, skip: Vec<Algorithm>) -> Self {
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
            .add_row(vec![
                "Sorting Metric".to_owned(),
                format!("{}", self.sort_by),
            ])
            .add_row(vec!["Shuffle Data".to_owned(), format!("{}", self.shuffle)])
            .add_row(vec![
                "Number of CV Folds".to_owned(),
                format!("{}", self.number_of_folds),
            ])
            .add_row(vec![
                "Skipped Algorithms".to_owned(),
                format!("{}", &skiplist[0..skiplist.len() - 1]),
            ]);

        write!(f, "{}\n", table)
    }
}
