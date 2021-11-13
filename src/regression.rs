//! Auto-ML for regression models

use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::{naive::dense_matrix::DenseMatrix, Matrix},
    linear::{
        elastic_net::{ElasticNet, ElasticNetParameters},
        lasso::{Lasso, LassoParameters},
        linear_regression::{LinearRegression, LinearRegressionParameters},
        ridge_regression::{RidgeRegression, RidgeRegressionParameters},
    },
    math::{distance::Distance, num::RealNumber},
    metrics::{
        mean_absolute_error::MeanAbsoluteError, mean_squared_error::MeanSquareError, r2::R2,
    },
    neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters},
    svm::{
        svr::{SVRParameters, SVR},
        Kernel,
    },
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};

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
pub struct ModelResult {
    model: Box<dyn Regressor>,
    r_squared: f32,
    mean_absolute_error: f32,
    mean_squared_error: f32,
    name: String,
}

trait Regressor {}
impl<T: RealNumber, M: Matrix<T>> Regressor for LinearRegression<T, M> {}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Regressor for SVR<T, M, K> {}
impl<T: RealNumber, M: Matrix<T>> Regressor for Lasso<T, M> {}
impl<T: RealNumber, M: Matrix<T>> Regressor for RidgeRegression<T, M> {}
impl<T: RealNumber, M: Matrix<T>> Regressor for ElasticNet<T, M> {}
impl<T: RealNumber> Regressor for DecisionTreeRegressor<T> {}
impl<T: RealNumber, D: Distance<Vec<T>, T>> Regressor for KNNRegressor<T, D> {}
impl<T: RealNumber> Regressor for RandomForestRegressor<T> {}

/// This is the output from a model comparison operation
pub struct ModelComparison(Vec<ModelResult>);

impl Display for ModelComparison {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "R^2", "MSE", "MAE"]);
        for model_results in &self.0 {
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

/// The settings artifact for all regressions
pub struct Settings {
    sort_by: SortBy,
    linear_settings: LinearRegressionParameters,
    // svr_settings: SVRParameters<f32, DenseMatrix<f32>, K>,
    lasso_settings: LassoParameters<f32>,
    ridge_settings: RidgeRegressionParameters<f32>,
    elastic_net_settings: ElasticNetParameters<f32>,
    decision_tree_settings: DecisionTreeRegressorParameters,
    random_forest_settings: RandomForestRegressorParameters,
    // knn_regression_settings: KNNRegressorParameters<f32, Euclidian>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: SortBy::RSquared,
            linear_settings: LinearRegressionParameters::default(),
            // svr_settings: SVRParameters::default(),
            lasso_settings: LassoParameters::default(),
            ridge_settings: RidgeRegressionParameters::default(),
            elastic_net_settings: ElasticNetParameters::default(),
            decision_tree_settings: DecisionTreeRegressorParameters::default(),
            random_forest_settings: RandomForestRegressorParameters::default(),
            // knn_regression_settings: KNNRegressorParameters::default(),
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
/// let data = smartcore::dataset::breast_cancer::load_dataset();
/// let settings = automl::regression::Settings::default().sorted_by(automl::regression::SortBy::MeanSquaredError);
/// let x = automl::regression::compare_models(data, settings);
/// print!("{}", x);
/// ```
pub fn compare_models(dataset: Dataset<f32, f32>, settings: Settings) -> ModelComparison {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let mut results = Vec::new();

    // Do the standard linear model
    let model = LinearRegression::fit(&x, &y, settings.linear_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Linear Regression".to_string(),
    });

    let model = SVR::fit(&x, &y, SVRParameters::default().with_eps(2.0).with_c(10.0)).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Support Vector Regression".to_string(),
    });

    let model = Lasso::fit(&x, &y, settings.lasso_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "LASSO".to_string(),
    });

    let model = RidgeRegression::fit(&x, &y, settings.ridge_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Ridge Regression".to_string(),
    });

    let model = ElasticNet::fit(&x, &y, settings.elastic_net_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Elastic Net".to_string(),
    });

    let model = DecisionTreeRegressor::fit(&x, &y, settings.decision_tree_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Decision Tree Regression".to_string(),
    });

    let model = RandomForestRegressor::fit(&x, &y, settings.random_forest_settings).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Random Forest Regression".to_string(),
    });

    let model = KNNRegressor::fit(&x, &y, KNNRegressorParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "KNN Regression".to_string(),
    });

    match settings.sort_by {
        SortBy::RSquared => {
            results.sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap_or(Equal));
        }
        SortBy::MeanSquaredError => {
            results.sort_by(|a, b| {
                a.mean_squared_error
                    .partial_cmp(&b.mean_squared_error)
                    .unwrap_or(Equal)
            });
        }
        SortBy::MeanAbsoluteError => {
            results.sort_by(|a, b| {
                a.mean_absolute_error
                    .partial_cmp(&b.mean_absolute_error)
                    .unwrap_or(Equal)
            });
        }
    }

    ModelComparison(results)
}
