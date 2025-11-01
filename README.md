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

Support Vector Regression can be enabled alongside the default algorithms and tuned with a
kernel-specific configuration:

```rust
use automl::settings::{Kernel, SVRParameters};
use automl::RegressionSettings;
use smartcore::linalg::basic::matrix::DenseMatrix;

let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
    RegressionSettings::default().with_svr_settings(
    SVRParameters::default()
        .with_eps(0.2)
        .with_tol(1e-4)
        .with_c(2.0)
        .with_kernel(Kernel::RBF(0.4)),
);
```

Gradient boosting via Smartcore's `XGBoost` implementation is also available, giving access to
learning-rate, depth, and subsampling knobs:

```rust
use automl::settings::XGRegressorParameters;
use automl::{DenseMatrix, RegressionSettings};

let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
    RegressionSettings::default().with_xgboost_settings(
    XGRegressorParameters::default()
        .with_n_estimators(75)
        .with_learning_rate(0.15)
        .with_max_depth(4)
        .with_subsample(0.9),
);
```

Extremely randomized trees offer another ensemble option that leans into randomness for lower
variance models:

```rust
use automl::settings::ExtraTreesRegressorParameters;
use automl::{DenseMatrix, RegressionSettings};

let settings: RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
    RegressionSettings::default().with_extra_trees_settings(
    ExtraTreesRegressorParameters::default()
        .with_n_trees(50)
        .with_min_samples_leaf(2)
        .with_keep_samples(true)
        .with_seed(7),
);
```

Unlike the random forest regressor, the Extra Trees variant grows each tree on the full training
set and samples split thresholds uniformly rather than optimizing them. The parameter
`with_keep_samples(true)` is particularly useful here: because there is no bootstrapping, enabling
it stores the original observations so that out-of-bag style diagnostics remain possible. You can
also adjust `with_m(...)` to change how many random features are considered at each split—doing so
directly influences the amount of randomness introduced by the split selection compared with the
random forest estimator.

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

Bernoulli Naive Bayes supports binary features and can also binarize continuous inputs when you
provide a threshold. Set `binarize` to `None` to require pre-binarized inputs, or configure the
threshold to map values above it to `1` and the rest to `0` during training and prediction:

```rust
use automl::settings::{BernoulliNBParameters, ClassificationSettings};

let mut params = BernoulliNBParameters::default();
params.binarize = None; // ensure features are already 0/1 encoded
let settings = ClassificationSettings::default().with_bernoulli_nb_settings(params);

// alternatively, binarize values greater than 0.5
let thresholded = ClassificationSettings::default().with_bernoulli_nb_settings(
    BernoulliNBParameters::default().with_binarize(0.5),
);
```

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

for algorithm in model.trained_algorithm_names() {
    let clusters: Vec<u8> = model.predict_with(algorithm, &x).expect("prediction");
    println!("{algorithm}: {clusters:?}");
}
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
- Regression: Decision Tree, KNN, Random Forest, Extra Trees, Linear, Ridge, LASSO, Elastic Net, Support Vector Regression, `XGBoost` Gradient Boosting
- Classification: Random Forest, Decision Tree, KNN, Logistic Regression, Support Vector Classifier, Gaussian Naive Bayes, Categorical Naive Bayes, Bernoulli Naive Bayes (binary features or configurable thresholding), Categorical Naive Bayes, Multinomial Naive Bayes (non-negative integer features)
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
