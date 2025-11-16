[![ci](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

# `automl` with `smartcore`

`AutoML` (_Automated Machine Learning_) streamlines machine learning workflows, making them more accessible and
efficient
for users of all experience levels. This crate extends the [`smartcore`](https://docs.rs/smartcore/) machine learning
framework, providing utilities to quickly train, compare, and deploy models.

# Install

Add `automl` to your `Cargo.toml` to get started:

**Stable Version**

```toml
automl = "0.3.0"
```

**Latest Development Version**

```toml
automl = { git = "https://github.com/cmccomb/rust-automl" }
```

# Example Usage

Here’s a quick example to illustrate how `AutoML` can simplify model training and comparison:

```rust, no_run, ignore
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

## Cookbook

Explore the `automl::cookbook` module for copy-pastable examples that mirror
real-world workflows:

- `cargo run --example breast_cancer_csv` – load the Wisconsin Diagnostic
  Breast Cancer dataset from CSV, standardize features, and compare tuned
  classifiers.
- `cargo run --example diabetes_regression` – impute, scale, and train
  regression models for the diabetes progression dataset.

## Preprocessing pipelines

`automl` now supports composable preprocessing pipelines so you can build
feature engineering recipes similar to `AutoGluon` or `caret`. Pipelines are
defined with the [`PreprocessingStep`](https://docs.rs/automl/latest/automl/settings/enum.PreprocessingStep.html)
enum and attached via either the `add_step` builder or by passing a full
[`PreprocessingPipeline`](https://docs.rs/automl/latest/automl/settings/struct.PreprocessingPipeline.html).

```rust
use automl::settings::{
    ClassificationSettings, PreprocessingPipeline, PreprocessingStep, RegressionSettings,
    StandardizeParams,
};
use automl::DenseMatrix;

let regression = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    .add_step(PreprocessingStep::Standardize(StandardizeParams::default()))
    .add_step(PreprocessingStep::ReplaceWithPCA {
        number_of_components: 5,
    });

let classification = ClassificationSettings::default().with_preprocessing(
    PreprocessingPipeline::new()
        .add_step(PreprocessingStep::AddInteractions)
        .add_step(PreprocessingStep::ReplaceWithSVD {
            number_of_components: 4,
        }),
);
```

Pipelines preserve the order of steps. Stateful steps such as PCA, SVD, or
standardization automatically fit during training and reuse the same fitted
state when you call `predict`.

### Supported preprocessing steps

The pipeline is intentionally modular so you can mix and match steps to
mirror popular `AutoML` defaults:

- **Scaling** – standard, min–max, or robust scaling via `ScaleParams`.
- **Imputation** – mean, median, or most-frequent replacement with
  `ImputeParams`.
- **Categorical encoders** – ordinal or one-hot encoding with optional dummy
  drop.
- **Power transforms** – per-column log or Box-Cox transforms with automatic
  shifting for strictly positive domains.
- **Column filters** – select or exclude features using `ColumnSelector`
  helpers.

Each state stores the fitted statistics (e.g., medians, category mappings) so
that the same transformation can be applied consistently during inference.

### Example: AutoGluon-style defaults

```rust, no_run
use automl::settings::{
    CategoricalEncoderParams, CategoricalEncoding, ColumnFilterParams, ColumnSelector,
    ImputeParams, ImputeStrategy, PowerTransform, PowerTransformParams, PreprocessingPipeline,
    PreprocessingStep, RobustScaleParams, ScaleParams, ScaleStrategy,
};

let pipeline = PreprocessingPipeline::new()
    // Fill numeric columns with the median before scaling.
    .add_step(PreprocessingStep::Impute(ImputeParams {
        strategy: ImputeStrategy::Median,
        selector: ColumnSelector::Include(vec![0, 1, 2]),
    }))
    // Apply a robust scaler to guard against outliers (similar to AutoGluon).
    .add_step(PreprocessingStep::Scale(ScaleParams {
        strategy: ScaleStrategy::Robust(RobustScaleParams::default()),
        selector: ColumnSelector::Include(vec![0, 1, 2]),
    }))
    // One-hot encode categorical columns and drop the reference level.
    .add_step(PreprocessingStep::EncodeCategorical(CategoricalEncoderParams {
        selector: ColumnSelector::Include(vec![3, 4]),
        encoding: CategoricalEncoding::one_hot(true),
    }))
    // Optionally keep only the engineered features.
    .add_step(PreprocessingStep::FilterColumns(ColumnFilterParams {
        selector: ColumnSelector::Include(vec![0, 1, 2, 5, 6]),
        retain_selected: true,
    }));
```

### Example: caret-style log + standardization recipe

```rust, no_run
use automl::settings::{
    ColumnSelector, PowerTransform, PowerTransformParams, PreprocessingPipeline,
    PreprocessingStep, ScaleParams, ScaleStrategy, StandardizeParams,
};

let caret_like = PreprocessingPipeline::new()
    .add_step(PreprocessingStep::PowerTransform(PowerTransformParams {
        selector: ColumnSelector::Include(vec![0]),
        transform: PowerTransform::Log { offset: 0.0 },
    }))
    .add_step(PreprocessingStep::Scale(ScaleParams {
        strategy: ScaleStrategy::Standard(StandardizeParams::default()),
        selector: ColumnSelector::All,
    }));
```

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
