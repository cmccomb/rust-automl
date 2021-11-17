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
let mut classifier = automl::classification::Classifier::default();
classifier.with_dataset(smartcore::dataset::breast_cancer::load_dataset());
classifier.compare_models();
print!("{}", classifier);
```
Will output this comparison of models usign cross-validation:
```text
┌────────────────────────────────┬───────────────────┬──────────────────┐
│ Model                          │ Training Accuracy │ Testing Accuracy │
╞════════════════════════════════╪═══════════════════╪══════════════════╡
│ Random Forest Classifier       │ 1.00              │ 0.96             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ Logistic Regression Classifier │ 0.97              │ 0.95             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ Gaussian Naive Bayes           │ 0.95              │ 0.93             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ KNN Classifier                 │ 0.96              │ 0.92             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ Categorical Naive Bayes        │ 0.96              │ 0.91             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ Decision Tree Classifier       │ 1.00              │ 0.90             │
├────────────────────────────────┼───────────────────┼──────────────────┤
│ Support Vector Classifier      │ 0.87              │ 0.85             │
└────────────────────────────────┴───────────────────┴──────────────────┘
```
You can then train a final model using `classifier.train_final_model()` and perform inference using that model with the `predict` method.

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