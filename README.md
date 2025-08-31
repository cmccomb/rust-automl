[![Github CI](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/automl/actions)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

# `AutoML` with `SmartCore`

`AutoML` (_Automated Machine Learning_) streamlines machine learning workflows, making them more accessible and efficient
for users of all experience levels. This crate extends the [`smartcore`](https://docs.rs/smartcore/) machine learning
framework, providing utilities to
quickly train, compare, and deploy models.

# Install

Add `AutoML` to your `Cargo.toml` to get started:

**Stable Version**

```toml
automl = "0.2.9"
```

**Latest Development Version**

```toml
automl = { git = "https://github.com/cmccomb/rust-automl" }
```

# Example Usage

Here’s a quick example to illustrate how `AutoML` can simplify model training and comparison:

```rust
use automl::{Settings, SupervisedModel};
use smartcore::linalg::basic::matrix::DenseMatrix;

let x = DenseMatrix::from_2d_vec(&vec![
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
])
.unwrap();
let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let settings = Settings::default_regression();
let mut regressor = SupervisedModel::new(x, y, settings);
regressor.train();
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

This crate has several features that add some additional methods.

| Feature | Description                                                                                             |
|:--------|:--------------------------------------------------------------------------------------------------------|
| `nd`    | Adds methods for predicting/reading data using [`ndarray`](https://crates.io/crates/ndarray).           |
| `csv`   | Adds methods for predicting/reading data from a .csv using [`polars`](https://crates.io/crates/polars). |

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

## Development

Before submitting changes, ensure the codebase is clean and secure:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
```
