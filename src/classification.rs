//! Auto-ML for classification models

// use comfy_table::{
//     modifiers::{UTF8_ROUND_CORNERS, UTF8_SOLID_INNER_BORDERS},
//     presets::UTF8_FULL,
//     Table,
// };
// use linfa::dataset::{AsTargets, DatasetBase, Records};
// use linfa::prelude::*;
// use linfa::traits::{Fit, Predict};
// use ndarray::{Array1, Array2, ArrayBase, Axis, Data, DataMut, Dim, Ix1, Ix2, OwnedRepr};
// use std::fmt::{Display, Formatter};
//
// use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
//
// /// This contains the results of a single model, including the model itself
// pub struct ModelResult<T> {
//     model: Box<T>,
//     name: String,
// }
//
// /// This is the output from a model comparison operation
// pub struct ModelComparison<T>(Vec<ModelResult<T>>);
//
// #[doc(hidden)]
// // pub trait Classifier {}
// // impl Classifier for FittedLogisticRegression<f64, C> {}
// // impl<F, L> Classifier for DecisionTree<F, L> {}
//
// ///
// pub fn compare_models<
//     F: Float + linfa_logistic::Float + std::cmp::Ord,
//     D: Data<Elem = F>,
//     T: AsTargets<Elem = F>,
//     TT,
// >(
//     dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
// ) -> ModelComparison<TT> {
//     let mut results: Vec<ModelResult<TT>> = Vec::new();
//
//     let model = LogisticRegression::default().fit(dataset).unwrap();
//     results.push(ModelResult {
//         model: Box::new(model),
//         name: "Logistic".to_string(),
//     });
//
//     ModelComparison(results)
// }
