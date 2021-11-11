use eval_metrics::regression::{mae, mse, rmse, rsq};
use linfa::dataset::{AsTargets, DatasetBase, Records};
use linfa::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_elasticnet::ElasticNet;
use linfa_linear::{FittedLinearRegression, Float, LinearRegression};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, DataMut, Dim, Ix1, Ix2, OwnedRepr};

pub struct Settings {
    cv: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { cv: 10 }
    }
}

pub struct ModelResult<F: Float> {
    models: Box<dyn Regressor>,
    r2: F,
}

pub trait Regressor {}
impl<F: Float> Regressor for FittedLinearRegression<F> {}
impl<F: Float> Regressor for ElasticNet<F> {}

///
/// ```
/// let data = linfa_datasets::diabetes();
/// let r = automl::regression::compare_models(&data);
/// println!("{:?}", r);
/// panic!();
/// ```
pub fn compare_models<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = F>>(
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
) -> Vec<ModelResult<F>> {
    let mut results: Vec<ModelResult<F>> = Vec::new();

    let model = LinearRegression::default().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        models: Box::new(model),
        r2: y.r2(dataset).unwrap(),
    });

    let model = ElasticNet::params().fit(dataset).unwrap();
    let y = model.predict(dataset);
    results.push(ModelResult {
        models: Box::new(model),
        r2: y.r2(dataset).unwrap(),
    });

    // let model2 = ElasticNet::ridge().fit(dataset).unwrap();
    // let r22 = model2.predict(dataset);
    //
    // let model2 = ElasticNet::params().fit(dataset).unwrap();
    // let r22 = model2.predict(dataset);

    results
}

// pub fn compare_models<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = F>, R: Records>(
//     dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
//     settings: Settings,
// ) -> Vec<Box<dyn Regressor>> {
//     let mut models: Vec<Box<dyn Regressor>> = Vec::new();
//     models.push(Box::new(LinearRegression::default().fit(dataset).unwrap()));
//     models.push(Box::new(ElasticNet::ridge().fit(dataset).unwrap()));
//     models.push(Box::new(ElasticNet::lasso().fit(dataset).unwrap()));
//
//     models
// }
