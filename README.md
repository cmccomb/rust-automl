[![ci](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://docs.rs/automl/badge.svg)](https://docs.rs/automl)

# `automl` with `smartcore`

Train, compare, and predict with machine-learning models in Rust, using
[SmartCore](https://docs.rs/smartcore/) estimators and configurable preprocessing.

## Install

```toml
automl = "0.3.1"
```

## Quickstart

Compare classifiers with cross-validation, inspect their scores, and predict with
the best model:

```rust
use automl::{ClassificationModel, ClassificationSettings, DenseMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows: Vec<Vec<f64>> = (0..24).map(|i| vec![f64::from(i)]).collect();
    let features = DenseMatrix::from_2d_vec(&rows)?;
    let labels: Vec<u32> = (0..24).map(|i| u32::from(i >= 12)).collect();
    let settings = ClassificationSettings::default().with_number_of_folds(3);
    let mut classifier = ClassificationModel::new(features, labels, settings);
    classifier.train()?;

    println!("{classifier}");
    let predictions = classifier.predict(DenseMatrix::from_2d_array(&[&[2.0], &[20.0]])?)?;
    println!("Predictions: {predictions:?}");
    Ok(())
}
```

A successful `train()` replaces the previous results. A failed retraining attempt
preserves the last successful model and fitted preprocessing; settings edits are
retained so you can correct them and retry.

## Models and preprocessing

- **Regression:** linear, ridge, lasso, elastic net, decision trees, random forests,
  extra trees, KNN, support vector regression, and `XGBoost`.
- **Classification:** decision trees, random forests, logistic regression, KNN,
  support vector classifiers, and naive Bayes variants.
- **Clustering:** K-means, DBSCAN, and agglomerative clustering.
- **Preprocessing:** scaling, imputation, categorical encoding, power transforms,
  column selection, PCA/SVD, interactions, and polynomial features.

Configure algorithms and pipelines through
[`settings`](https://docs.rs/automl/latest/automl/settings/index.html).
Fitted preprocessing is reused for inference. See the
[cookbook](https://docs.rs/automl/latest/automl/cookbook/index.html) for pipeline
recipes and complete workflows.

## Examples and CSV input

```sh
cargo run --example breast_cancer_csv
cargo run --example diabetes_regression
```

CSV loading is available without feature flags through
[`load_csv_features`](https://docs.rs/automl/latest/automl/utils/fn.load_csv_features.html)
and [`load_labeled_csv`](https://docs.rs/automl/latest/automl/utils/fn.load_labeled_csv.html).

[API documentation](https://docs.rs/automl) · [Changelog](CHANGELOG.md)
