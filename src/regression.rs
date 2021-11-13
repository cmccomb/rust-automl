//! Auto-ML for regression models

use super::traits::Regressor;
use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
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

/// This is the output from a model comparison operation
pub struct ComparisonResults {
    results: Vec<Model>,
    sort_by: SortBy,
}

impl ComparisonResults {
    /// Uses the best model to make a prediction
    pub fn predict_with_best_model(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.results[0].name.as_str() {
            "Linear Regressor" => {
                let model: LinearRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "LASSO Regressor" => {
                let model: Lasso<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Ridge Regressor" => {
                let model: RidgeRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Elastic Net Regressor" => {
                let model: ElasticNet<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Random Forest Regressor" => {
                let model: RandomForestRegressor<f32> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "KNN Regressor" => {
                let model: KNNRegressor<f32, Euclidian> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Support Vector Regressor" => {
                let model: SVR<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Decision Tree Regressor" => {
                let model: DecisionTreeRegressor<f32> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            &_ => panic!("Unable to predict"),
        }
    }

    /// Returns a serialized version of the best model
    pub fn get_best_model(&self) -> Vec<u8> {
        self.results[0].model.clone()
    }

    fn add_model(&mut self, name: String, y_test: &Vec<f32>, y_pred: &Vec<f32>, model: Vec<u8>) {
        self.results.push(Model {
            model,
            r_squared: R2 {}.get_score(y_test, y_pred),
            mean_absolute_error: MeanAbsoluteError {}.get_score(y_test, y_pred),
            mean_squared_error: MeanSquareError {}.get_score(y_test, y_pred),
            name,
        });
        self.sort()
    }

    fn sort(&mut self) {
        match self.sort_by {
            SortBy::RSquared => {
                self.results
                    .sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap_or(Equal));
            }
            SortBy::MeanSquaredError => {
                self.results.sort_by(|a, b| {
                    a.mean_squared_error
                        .partial_cmp(&b.mean_squared_error)
                        .unwrap_or(Equal)
                });
            }
            SortBy::MeanAbsoluteError => {
                self.results.sort_by(|a, b| {
                    a.mean_absolute_error
                        .partial_cmp(&b.mean_absolute_error)
                        .unwrap_or(Equal)
                });
            }
        }
    }
    fn new(sort_by: SortBy) -> Self {
        Self {
            results: Vec::new(),
            sort_by,
        }
    }
}

impl Display for ComparisonResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "R^2", "MSE", "MAE"]);
        for model_results in &self.results {
            table.add_row(vec![
                format!("{}", &model_results.name),
                format!("{:.3}", &model_results.r_squared),
                format!("{:.3e}", &model_results.mean_squared_error),
                format!("{:.3e}", &model_results.mean_absolute_error),
            ]);
        }
        write!(f, "{}\n", table)
    }
}

/// An enum for sorting
#[non_exhaustive]
pub enum SortBy {
    /// Sort by R^2
    RSquared,
    /// Sort by MAE
    MeanAbsoluteError,
    /// Sort by MSE
    MeanSquaredError,
}

/// This contains the results of a single model, including the model itself
struct Model {
    model: Vec<u8>,
    r_squared: f32,
    mean_absolute_error: f32,
    mean_squared_error: f32,
    name: String,
}

/// The settings artifact for all regressions
pub struct Settings {
    sort_by: SortBy,
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
            sort_by: SortBy::RSquared,
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
    /// Adds a specific sorting function to the settings
    pub fn sorted_by(mut self, sort_by: SortBy) -> Self {
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

/// This function compares all of the regression models available in the package.
/// ```
/// let data = smartcore::dataset::diabetes::load_dataset();
/// let settings = automl::regression::Settings::default()
///     .sorted_by(automl::regression::SortBy::MeanSquaredError)
///     .with_svr_settings(smartcore::svm::svr::SVRParameters::default().with_eps(2.0).with_c(10.0));
/// let x = automl::regression::compare_models(data, settings);
/// print!("{}", x);
/// ```
pub fn compare_models(dataset: Dataset<f32, f32>, settings: Settings) -> ComparisonResults {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let (x_test, x_train, y_test, y_train) =
        train_test_split(&x, &y, settings.testing_fraction, settings.shuffle);

    let mut results = ComparisonResults::new(settings.sort_by);

    // Do the standard linear model
    let model = LinearRegression::fit(&x_train, &y_train, settings.linear_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model = SVR::fit(&x_train, &y_train, settings.svr_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model = Lasso::fit(&x_train, &y_train, settings.lasso_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model = RidgeRegression::fit(&x_train, &y_train, settings.ridge_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model = ElasticNet::fit(&x_train, &y_train, settings.elastic_net_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model =
        DecisionTreeRegressor::fit(&x_train, &y_train, settings.decision_tree_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model =
        RandomForestRegressor::fit(&x_train, &y_train, settings.random_forest_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model = KNNRegressor::fit(&x_train, &y_train, KNNRegressorParameters::default()).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    results
}
