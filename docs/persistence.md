# Model persistence (0.3.2)

Train a regressor, call `model.save("model.json")`, and restore it with
`RegressionModel::<f64, f64, DenseMatrix<f64>, Vec<f64>>::load("model.json")`.
Use matching input/output numeric and container types when loading.
The runnable `diabetes_regression` example demonstrates the complete workflow.

## Supported state

Linear, ridge, lasso, elastic-net, decision-tree, random-forest, extra-trees,
KNN, and support-vector regressors preserve their predictions. All fitted
preprocessing steps travel with the selected estimator, including category maps,
imputation statistics, scalers, PCA/SVD projections, and generated features.
SVR supports linear, polynomial, RBF, and sigmoid kernels, including models
whose prediction is only the fitted bias. XGBoost in SmartCore 0.4.2 does not
expose the required serialized state and returns `UnsupportedAlgorithm`.
Classification and clustering persistence are outside this release.

The artifact stores one selected estimator, fitted preprocessing, and the current
public settings. Settings may have been edited since the last successful training;
they are descriptive configuration, not proof of the estimator's training setup.
A failed retraining attempt preserves the prior estimator and preprocessing.
The wrapper's training buffers, cross-validation scores, and leaderboard are
omitted. Loaded models report `has_training_data() == false`, reject `train()`,
and support both prediction and saving again.

## Format and handling

Format version 1 explicitly records the SmartCore model version (0.4.2).
The loader validates these headers and supported fitted-state shapes before
deserializing. This is a versioned inference artifact, not a promise that future
SmartCore layouts will remain compatible. JSON uses exact float round trips so
numeric category keys survive reload unchanged.

A save writes a temporary file in the destination directory and atomically
replaces the destination after successful serialization. Unsupported models are
rejected before touching the destination. New Unix artifacts use mode 0600;
replacement retains the destination's existing permissions. Parent directories
must already exist. I/O and format failures are returned as `PersistenceError`.

Load trusted artifacts only. Structural checks are not a security boundary for
adversarial files. Estimators can retain training-derived data: KNN stores example
rows and targets, and SVR stores support vectors. Omitting wrapper training buffers
does not anonymize an artifact.

## Validation

`cargo test --test model_persistence` covers the nine families, f32 and f64,
all four SVR kernels, four KNN distances and both neighbor search strategies,
preprocessing variants, re-saving, failed retraining, malformed/version-mismatched
artifacts, and replacement behavior. Unix permission checks are platform-specific.
