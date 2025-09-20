[![Github CI](https://github.com/cmccomb/rust-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/automl/actions)
[![Crates.io](https://img.shields.io/crates/v/automl.svg)](https://crates.io/crates/automl)
[![docs.rs](https://img.shields.io/docsrs/automl/latest?logo=rust)](https://docs.rs/automl)

## What & Why
`automl` automates model selection and training on top of the `smartcore` machine learning library, helping Rust developers quickly build regression, classification, and clustering models.

## Quickstart
Install from [crates.io](https://crates.io/crates/automl) or use the GitHub repository for the latest changes:

```toml
# Cargo.toml
[dependencies]
automl = "0.2.9"
```

```toml
# Cargo.toml
[dependencies]
automl = { git = "https://github.com/cmccomb/rust-automl" }
```

```rust
use automl::{RegressionModel, RegressionSettings};
use smartcore::linalg::basic::matrix::DenseMatrix;

let x = DenseMatrix::from_2d_vec(&vec![
    vec![1.0_f64, 2.0, 3.0],
    vec![2.0, 3.0, 4.0],
    vec![3.0, 4.0, 5.0],
]).unwrap();
let y = vec![1.0_f64, 2.0, 3.0];
let _model = RegressionModel::new(x, y, RegressionSettings::default());
```

### Loading data from CSV

Use `load_labeled_csv` to read a dataset and separate the target column:

```rust
use automl::{RegressionModel, RegressionSettings};
use automl::utils::load_labeled_csv;

let (x, y) = load_labeled_csv("tests/fixtures/supervised_sample.csv", 2).unwrap();
let mut model = RegressionModel::new(x, y, RegressionSettings::default());
```

Use `load_csv_features` to read unlabeled data for clustering:

```rust
use automl::{ClusteringModel};
use automl::settings::ClusteringSettings;
use automl::utils::load_csv_features;

let x = load_csv_features("tests/fixtures/clustering_points.csv").unwrap();
let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
model.train();
let clusters: Vec<u8> = model.predict(&x).unwrap();
```

## Examples
### Classification
```rust
use automl::{ClassificationModel};
use automl::settings::{ClassificationSettings, RandomForestClassifierParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

let x = DenseMatrix::from_2d_vec(&vec![
    vec![0.0_f64, 0.0],
    vec![1.0, 1.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
]).unwrap();
let y = vec![0_u32, 1, 1, 0];
let settings = ClassificationSettings::default()
    .with_random_forest_classifier_settings(
        RandomForestClassifierParameters::default().with_n_trees(10),
    );
let _model = ClassificationModel::new(x, y, settings);
```

Multinomial Naive Bayes is available for datasets where every feature represents a non-negative
integer count. You can opt into it alongside the other classifiers when your data meets that
requirement:

```rust
use automl::settings::{ClassificationSettings, MultinomialNBParameters};

let settings = ClassificationSettings::default()
    .with_multinomial_nb_settings(MultinomialNBParameters::default());
```

If the feature matrix includes fractional or negative values, the Multinomial NB variant will
emit a descriptive error explaining the constraint.

### Clustering
```rust
use automl::ClusteringModel;
use automl::settings::ClusteringSettings;
use smartcore::linalg::basic::matrix::DenseMatrix;

let x = DenseMatrix::from_2d_vec(&vec![
    vec![1.0_f64, 1.0],
    vec![1.2, 0.8],
    vec![8.0, 8.0],
    vec![8.2, 8.2],
]).unwrap();
let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
model.train();
let truth = vec![1_u8, 1, 2, 2];
model.evaluate(&truth);
println!("{model}");
let _clusters: Vec<u8> = model.predict(&x).expect("prediction");
```

Additional runnable examples are available in the [examples/ directory](https://github.com/cmccomb/rust-automl/tree/main/examples),
including [minimal_classification.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/minimal_classification.rs),
[maximal_classification.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/maximal_classification.rs),
[minimal_regression.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/minimal_regression.rs),
[maximal_regression.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/maximal_regression.rs),
[minimal_clustering.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/minimal_clustering.rs), and
[maximal_clustering.rs](https://github.com/cmccomb/rust-automl/blob/main/examples/maximal_clustering.rs).

Model comparison:

```text
┌───────────────────────────────┬─────────────────────┬───────────────────┬──────────────────┐
│ Model                         │ Time                │ Training Accuracy │ Testing Accuracy │
╞═══════════════════════════════╪═════════════════════╪═══════════════════╪══════════════════╡
│ Random Forest Classifier      │ 835ms 393us 583ns   │ 1.00              │ 0.96             │
├───────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ Decision Tree Classifier      │ 15ms 404us 750ns    │ 1.00              │ 0.93             │
├───────────────────────────────┼─────────────────────┼───────────────────┼──────────────────┤
│ KNN Classifier                │ 28ms 874us 208ns    │ 0.96              │ 0.92             │
└───────────────────────────────┴─────────────────────┴───────────────────┴──────────────────┘
```

## Capabilities
- Feature Engineering: PCA, SVD, interaction terms, polynomial terms
- Regression: Decision Tree, KNN, Random Forest, Linear, Ridge, LASSO, Elastic Net, Support Vector Regression
- Classification: Random Forest, Decision Tree, KNN, Logistic Regression, Support Vector Classifier, Gaussian Naive Bayes, Categorical Naive Bayes, Multinomial Naive Bayes (non-negative integer features)
- Clustering: K-Means, Agglomerative, DBSCAN
- Meta-learning: Blending (experimental)
- Persistence: Save/load settings and models

## Development
Before submitting changes, run:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
cargo test --doc
```

Security audits run weekly via a scheduled workflow, but running `cargo audit` locally before submitting changes helps catch issues earlier.

Pull requests are welcome!

## License
Licensed under the MIT OR Apache-2.0 license.
