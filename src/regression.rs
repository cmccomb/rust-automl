//! Auto-ML for regression models

use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use linfa::dataset::{AsTargets, DatasetBase};
use linfa::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_elasticnet::ElasticNet;
use linfa_linear::{FittedLinearRegression, Float, LinearRegression};
use ndarray::{ArrayBase, Data, Ix2};
use std::fmt::{Display, Formatter};

/// This contains the results of a single model, including the model itself
pub struct ModelResult<F: Float> {
    model: Box<dyn Regressor>,
    r_squared: F,
    mean_absolute_error: F,
    mean_squared_error: F,
    explained_variance: F,
    name: String,
}

/// This is the output from a model comparison operation
pub struct ModelComparison<F: Float>(Vec<ModelResult<F>>);

impl<F: Float> Display for ModelComparison<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "R^2", "MSE", "MAE", "Exp. Var."]);
        for model_results in &self.0 {
            table.add_row(vec![
                format!("{}", &model_results.name),
                format!("{:.3}", &model_results.r_squared),
                format!("{:.3e}", &model_results.mean_squared_error),
                format!("{:.3e}", &model_results.mean_absolute_error),
                format!("{:.3e}", &model_results.explained_variance),
            ]);
        }
        write!(f, "{}\n", table)
    }
}

#[doc(hidden)]
pub trait Regressor {}
impl<F: Float> Regressor for FittedLinearRegression<F> {}
impl<F: Float> Regressor for ElasticNet<F> {}

///
/// ```
/// let data = linfa_datasets::diabetes();
/// let r = automl::regression::compare_models(&data);
/// ```
pub fn compare_models<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = F>>(
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
) -> ModelComparison<F> {
    let mut results: Vec<ModelResult<F>> = Vec::new();

    let model = LinearRegression::default().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: y.r2(dataset).unwrap(),
        mean_absolute_error: y.mean_absolute_error(dataset).unwrap(),
        mean_squared_error: y.mean_squared_error(dataset).unwrap(),
        explained_variance: y.explained_variance(dataset).unwrap(),
        name: "Linear Model".to_string(),
    });

    let model = ElasticNet::params().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: y.r2(dataset).unwrap(),
        mean_absolute_error: y.mean_absolute_error(dataset).unwrap(),
        mean_squared_error: y.mean_squared_error(dataset).unwrap(),
        explained_variance: y.explained_variance(dataset).unwrap(),
        name: "Elastic Net".to_string(),
    });

    let model = ElasticNet::lasso().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: y.r2(dataset).unwrap(),
        mean_absolute_error: y.mean_absolute_error(dataset).unwrap(),
        mean_squared_error: y.mean_squared_error(dataset).unwrap(),
        explained_variance: y.explained_variance(dataset).unwrap(),
        name: "LASSO".to_string(),
    });

    let model = ElasticNet::ridge().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        model: Box::new(model),
        r_squared: y.r2(dataset).unwrap(),
        mean_absolute_error: y.mean_absolute_error(dataset).unwrap(),
        mean_squared_error: y.mean_squared_error(dataset).unwrap(),
        explained_variance: y.explained_variance(dataset).unwrap(),
        name: "Ridge".to_string(),
    });

    ModelComparison(results)
}
