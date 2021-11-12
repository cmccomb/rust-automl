//! Auto-ML for regression models

use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::{naive::dense_matrix::DenseMatrix, Matrix},
    linear::{
        elastic_net::{ElasticNet, ElasticNetParameters},
        lasso::{Lasso, LassoParameters},
        linear_regression::{
            LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
        },
        ridge_regression::{RidgeRegression, RidgeRegressionParameters, RidgeRegressionSolverName},
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

/// This function compares all of the regression models available in the package.
/// ```
/// let data = smartcore::dataset::breast_cancer::load_dataset();
/// let x = automl::regression::compare_models(data);
/// print!("{}", x);
/// panic!()
/// ```
pub fn compare_models(dataset: Dataset<f32, f32>) -> ModelComparison {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let mut results = Vec::new();

    // Do the standard linear model
    let model = LinearRegression::fit(&x, &y, LinearRegressionParameters::default()).unwrap();
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

    let model = Lasso::fit(&x, &y, LassoParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "LASSO".to_string(),
    });

    let model = RidgeRegression::fit(&x, &y, RidgeRegressionParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Ridge Regression".to_string(),
    });

    let model = ElasticNet::fit(&x, &y, ElasticNetParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Elastic Net".to_string(),
    });

    let model =
        DecisionTreeRegressor::fit(&x, &y, DecisionTreeRegressorParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: R2 {}.get_score(&y, &y_pred),
        mean_absolute_error: MeanAbsoluteError {}.get_score(&y, &y_pred),
        mean_squared_error: MeanSquareError {}.get_score(&y, &y_pred),
        name: "Decision Tree Regression".to_string(),
    });

    let model =
        RandomForestRegressor::fit(&x, &y, RandomForestRegressorParameters::default()).unwrap();
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

    results.sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap_or(Equal));

    ModelComparison(results)
}
