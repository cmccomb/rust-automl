[![Github CI](https://github.com/cmccomb/rust-automl/actions/workflows/tests.yml/badge.svg)](https://github.com/cmccomb/automl/actions)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

# AutoML with SmartCore
AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accessible for 
a general audience. This crate builds on top of the [smartcore](https://smartcorelib.org/) machine learning framework, 
and provides some utilities to quickly train and compare models. 

# Usage
For instance, running the following:
```rust
fn main() {
  let dataset = smartcore::dataset::diabetes::load_dataset();
  let settings = automl::regression::Settings::default();
  let mut regressor = automl::regression::Regressor::new(settings);
  regressor.with_dataset(dataset);
  regressor.compare_models();
  print!("{}", regressor);
}
```
Will output this:
```text
┌──────────────────────────┬──────────────┬─────────────┐
│ Model                    │ Training R^2 │ Testing R^2 │
╞══════════════════════════╪══════════════╪═════════════╡
│ LASSO Regressor          │ 0.52         │ 0.49        │
├──────────────────────────┼──────────────┼─────────────┤
│ Linear Regressor         │ 0.52         │ 0.48        │
├──────────────────────────┼──────────────┼─────────────┤
│ Ridge Regressor          │ 0.52         │ 0.47        │
├──────────────────────────┼──────────────┼─────────────┤
│ Elastic Net Regressor    │ 0.47         │ 0.45        │
├──────────────────────────┼──────────────┼─────────────┤
│ Random Forest Regressor  │ 0.90         │ 0.40        │
├──────────────────────────┼──────────────┼─────────────┤
│ KNN Regressor            │ 0.66         │ 0.29        │
├──────────────────────────┼──────────────┼─────────────┤
│ Support Vector Regressor │ -0.01        │ -0.03       │
├──────────────────────────┼──────────────┼─────────────┤
│ Decision Tree Regressor  │ 1.00         │ -0.17       │
└──────────────────────────┴──────────────┴─────────────┘
```
Based on this output, you can then select the best model for the task.

## Features
Currently this crate only has AutoML features for regression and classification. This includes the following models:
- Regression
  - Decision Tree Regression
  - KNN Regression
  - Random Forest Regression
  - Linear Regression
  - Rdige Regression
  - LASSO
  - Elastic Net
  - Support Vector Regression
- Classification
  - Random Forest Classification
  - Decision Tree Classification
  - Support Vector Classification
  - Logistic Regression
  - KNN Classification