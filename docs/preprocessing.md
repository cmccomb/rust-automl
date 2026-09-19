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
