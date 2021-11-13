[![Github CI](https://github.com/cmccomb/rust-automl/actions/workflows/tests.yml/badge.svg)](https://github.com/cmccomb/automl/actions)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

# AutoML with SmartCore
AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accessible for 
a general audience. This crate builds on top of the [smartcore](https://smartcorelib.org/) machine learning framework, 
and provides some utilities to quickly train and compare models. 

# Usage
Running the following:
For instance, running the following:
```rust
fn main() {
  let data = smartcore::dataset::breast_cancer::load_dataset();
  let settings = automl::regression::Settings::default();
  let results = automl::regression::compare_models(data, settings);
  print!("{}", results);
}
```
Will output this:
```text
┌───────────────────────────┬────────┬───────────┬──────────┐
│ Model                     │ R^2    │ MSE       │ MAE      │
╞═══════════════════════════╪════════╪═══════════╪══════════╡
│ Decision Tree Regression  │ 1.000  │ 1.638e-12 │ 5.531e-8 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ Random Forest Regression  │ 0.972  │ 6.626e-3  │ 2.830e-2 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ KNN Regression            │ 0.878  │ 2.851e-2  │ 5.624e-2 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ Linear Regression         │ 0.773  │ 5.309e-2  │ 1.813e-1 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ Ridge Regression          │ 0.772  │ 5.320e-2  │ 1.822e-1 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ Elastic Net               │ 0.385  │ 1.437e-1  │ 3.591e-1 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ LASSO                     │ 0.000  │ 2.338e-1  │ 4.675e-1 │
├───────────────────────────┼────────┼───────────┼──────────┤
│ Support Vector Regression │ -0.069 │ 2.500e-1  │ 5.000e-1 │
└───────────────────────────┴────────┴───────────┴──────────┘
```
Based on this output, you can then select the best model for the task.

## Features
Currently this crate only includes AutoML features for regression and classification. 
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
  - Support Vector Classification
  - Logistic Regression
  - KNN Classification