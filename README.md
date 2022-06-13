[![Github CI](https://github.com/cmccomb/rust-automl/actions/workflows/tests.yml/badge.svg)](https://github.com/cmccomb/automl/actions)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

# AutoML with SmartCore
AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accessible for 
a general audience. This crate builds on top of the [smartcore](https://docs.rs/smartcore/) machine learning framework, 
and provides some utilities to quickly train and compare models. 

# Install
To use the latest released version of `AutoML`, add this to your `Cargo.toml`:
```toml
automl = "0.2.6"
```
To use the bleeding edge instead, add this:
```toml
automl = { git = "https://github.com/cmccomb/rust-automl" }
```

# Usage
Running the following:
```rust
let dataset = smartcore::dataset::breast_cancer::load_dataset();
let settings = automl::Settings::default_classification();
let mut classifier = automl::SupervisedModel::new(dataset, settings);
classifier.train();
```
will perform a comparison of classifier models using cross-validation. Printing the classifier object will yield:
```text
┌────────────────────────────────┬─────────────────────┬───────────────────┬──────────────────┐
│ Model                          │ Time                │ Training Accuracy │ Testing Accuracy │
╞════════════════════════════════╪═════════════════════╪═══════════════════╪══════════════════╡
│ Random Forest Classifier       │ 835ms 393us 583ns   │ 1.00              │ 0.96             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Logistic Regression Classifier │ 620ms 714us 583ns   │ 0.97              │ 0.95             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Gaussian Naive Bayes           │ 6ms 529us           │ 0.94              │ 0.93             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Categorical Naive Bayes        │ 2ms 922us 250ns     │ 0.96              │ 0.93             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Decision Tree Classifier       │ 15ms 404us 750ns    │ 1.00              │ 0.93             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ KNN Classifier                 │ 28ms 874us 208ns    │ 0.96              │ 0.92             │
├────────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Support Vector Classifier      │ 4s 187ms 61us 708ns │ 0.57              │ 0.57             │
└────────────────────────────────┴─────────────────────┴───────────────────┴──────────────────┘
```
You can then perform inference using the best model with the `predict` method.

## Features
This crate has several features that add some additional methods

| Feature   | Description                                                                                               |
|:----------|:----------------------------------------------------------------------------------------------------------|
| `nd`      | Adds methods for predicting/reading data using [`ndarray`](https://crates.io/crates/ndarray).             |
| `csv`     | Adds methods for predicting/reading data from a .csv using [`polars`](https://crates.io/crates/polars).   |

## Capabilities
- Feature Engineering
  - PCA 
  - SVD
  - Interaction terms
  - Polynomial terms
- Regression
  - Decision Tree Regression
  - KNN Regression
  - Random Forest Regression
  - Linear Regression
  - Ridge Regression
  - LASSO
  - Elastic Net
  - Support Vector Regression
- Classification
  - Random Forest Classification
  - Decision Tree Classification
  - Support Vector Classification
  - Logistic Regression
  - KNN Classification
  - Gaussian Naive Bayes
- Meta-learning
  - Blending
- Save and load settings
- Save and load models