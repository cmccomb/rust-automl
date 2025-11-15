#![allow(clippy::float_cmp)]

use automl::DenseMatrix;
use automl::model::preprocessing::Preprocessor;
use automl::settings::{
    CategoricalEncoderParams, CategoricalEncoding, ColumnFilterParams, ColumnSelector,
    ImputeParams, ImputeStrategy, MinMaxParams, PowerTransform, PowerTransformParams,
    PreprocessingPipeline, PreprocessingStep, RobustScaleParams, ScaleParams, ScaleStrategy,
    StandardizeParams,
};
use smartcore::linalg::basic::arrays::Array;

fn build_matrix(data: Vec<Vec<f64>>) -> DenseMatrix<f64> {
    let matrix = DenseMatrix::from_2d_vec(&data).expect("valid matrix");
    drop(data);
    matrix
}

#[test]
fn min_max_scaling_applies_range() {
    let x = build_matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let params = ScaleParams {
        strategy: ScaleStrategy::MinMax(MinMaxParams::default()),
        selector: ColumnSelector::All,
    };
    let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Scale(params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x.clone(), &pipeline).unwrap();
    assert_eq!(transformed.shape(), (2, 2));
    assert_eq!(*transformed.get((0, 0)), 0.0);
    assert_eq!(*transformed.get((1, 0)), 1.0);
    let reapplied = preprocessor.preprocess(x).unwrap();
    assert_eq!(*reapplied.get((0, 1)), 0.0);
    assert_eq!(*reapplied.get((1, 1)), 1.0);
}

#[test]
fn robust_scaling_centers_by_median() {
    let x = build_matrix(vec![vec![1.0], vec![100.0], vec![3.0]]);
    let params = ScaleParams {
        strategy: ScaleStrategy::Robust(RobustScaleParams::default()),
        selector: ColumnSelector::All,
    };
    let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Scale(params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x, &pipeline).unwrap();
    // Median is 3, IQR roughly 97
    assert!((*transformed.get((1, 0))).abs() > 0.9);
    assert!((*transformed.get((0, 0))).abs() < 1.0);
}

#[test]
fn imputer_fills_missing_mean() {
    let nan = f64::NAN;
    let x = build_matrix(vec![vec![1.0], vec![nan], vec![3.0]]);
    let params = ImputeParams {
        strategy: ImputeStrategy::Mean,
        selector: ColumnSelector::All,
    };
    let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::Impute(params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x.clone(), &pipeline).unwrap();
    assert_eq!(*transformed.get((1, 0)), 2.0);
    let reapplied = preprocessor.preprocess(x).unwrap();
    assert_eq!(*reapplied.get((1, 0)), 2.0);
}

#[test]
fn categorical_encoding_handles_one_hot_and_ordinal() {
    let ordinal = build_matrix(vec![vec![10.0, 0.0], vec![20.0, 1.0]]);
    let ordinal_params = CategoricalEncoderParams {
        selector: ColumnSelector::Include(vec![0]),
        encoding: CategoricalEncoding::Ordinal,
    };
    let pipeline =
        PreprocessingPipeline::new().add_step(PreprocessingStep::EncodeCategorical(ordinal_params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor
        .fit_transform(ordinal.clone(), &pipeline)
        .unwrap();
    assert_eq!(*transformed.get((0, 0)), 0.0);
    assert_eq!(*transformed.get((1, 0)), 1.0);

    let one_hot_params = CategoricalEncoderParams {
        selector: ColumnSelector::Include(vec![1]),
        encoding: CategoricalEncoding::one_hot(false),
    };
    let pipeline =
        PreprocessingPipeline::new().add_step(PreprocessingStep::EncodeCategorical(one_hot_params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(ordinal, &pipeline).unwrap();
    assert_eq!(transformed.shape(), (2, 3));
    assert_eq!(*transformed.get((0, 2)), 0.0);
    assert_eq!(*transformed.get((1, 2)), 1.0);
}

#[test]
fn power_transforms_apply_log_and_box_cox() {
    let x = build_matrix(vec![vec![2.0], vec![4.0]]);
    let log_params = PowerTransformParams {
        selector: ColumnSelector::All,
        transform: PowerTransform::Log { offset: 0.0 },
    };
    let pipeline =
        PreprocessingPipeline::new().add_step(PreprocessingStep::PowerTransform(log_params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x.clone(), &pipeline).unwrap();
    assert!(
        (*transformed.get((1, 0)) - (*transformed.get((0, 0)) + (4.0f64.ln() - 2.0f64.ln()))).abs()
            < 1e-9
    );

    let box_cox_params = PowerTransformParams {
        selector: ColumnSelector::All,
        transform: PowerTransform::BoxCox { lambda: 0.5 },
    };
    let pipeline =
        PreprocessingPipeline::new().add_step(PreprocessingStep::PowerTransform(box_cox_params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x, &pipeline).unwrap();
    assert!(transformed.get((0, 0)).is_finite());
    assert!(transformed.get((1, 0)).is_finite());
}

#[test]
fn column_filter_drops_unwanted_features() {
    let x = build_matrix(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let params = ColumnFilterParams {
        selector: ColumnSelector::Include(vec![0, 2]),
        retain_selected: true,
    };
    let pipeline = PreprocessingPipeline::new().add_step(PreprocessingStep::FilterColumns(params));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x, &pipeline).unwrap();
    assert_eq!(transformed.shape(), (2, 2));
    assert_eq!(*transformed.get((0, 1)), 3.0);
}

#[test]
fn pipelines_support_composed_steps() {
    let nan = f64::NAN;
    let x = build_matrix(vec![vec![nan, 10.0], vec![2.0, 20.0]]);
    let pipeline = PreprocessingPipeline::new()
        .add_step(PreprocessingStep::Impute(ImputeParams {
            strategy: ImputeStrategy::Median,
            selector: ColumnSelector::Include(vec![0]),
        }))
        .add_step(PreprocessingStep::Scale(ScaleParams {
            strategy: ScaleStrategy::Standard(StandardizeParams::default()),
            selector: ColumnSelector::All,
        }))
        .add_step(PreprocessingStep::EncodeCategorical(
            CategoricalEncoderParams {
                selector: ColumnSelector::Include(vec![1]),
                encoding: CategoricalEncoding::one_hot(true),
            },
        ));
    let mut preprocessor = Preprocessor::<f64, DenseMatrix<f64>>::new();
    let transformed = preprocessor.fit_transform(x, &pipeline).unwrap();
    assert_eq!(transformed.shape(), (2, 2));
    assert!((*transformed.get((0, 0))).abs() <= 1.0);
    assert_eq!(*transformed.get((1, 1)), 1.0);
}
