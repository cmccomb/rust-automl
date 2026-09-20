#[path = "fixtures/regression_data.rs"]
mod regression_data;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use automl::algorithms::RegressionAlgorithm;
use automl::settings::{
    ColumnSelector, ImputeParams, ImputeStrategy, KNNParameters, Kernel, PreprocessingPipeline,
    PreprocessingStep, RegressionSettings, SVRParameters, ScaleParams, ScaleStrategy,
    StandardizeParams, XGRegressorParameters,
};
use automl::{DenseMatrix, PersistenceError, RegressionModel};
use regression_data::regression_testing_data;

type Algorithm = RegressionAlgorithm<f64, f64, DenseMatrix<f64>, Vec<f64>>;
type Model = RegressionModel<f64, f64, DenseMatrix<f64>, Vec<f64>>;
type Settings = RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>;

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

struct TempModelFile(PathBuf);

impl TempModelFile {
    fn new(label: &str) -> Self {
        let id = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("automl-{label}-{}-{id}.json", std::process::id())))
    }
}

impl Drop for TempModelFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn inference_rows() -> DenseMatrix<f64> {
    DenseMatrix::from_2d_array(&[
        &[234.289, 235.6, 159.0, 107.608, 1947.0, 60.323],
        &[259.426, 232.5, 145.6, 108.632, 1948.0, 61.122],
    ])
    .unwrap()
}

fn assert_predictions_match(expected: &[f64], actual: &[f64]) {
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual) {
        assert!(
            (expected - actual).abs() <= 1e-10,
            "expected {expected}, got {actual}"
        );
    }
}

fn json_contains_number(value: &serde_json::Value, expected: f64) -> bool {
    match value {
        serde_json::Value::Number(number) => number
            .as_f64()
            .is_some_and(|actual| (actual - expected).abs() <= f64::EPSILON),
        serde_json::Value::Array(values) => values
            .iter()
            .any(|value| json_contains_number(value, expected)),
        serde_json::Value::Object(entries) => entries
            .values()
            .any(|value| json_contains_number(value, expected)),
        _ => false,
    }
}

fn assert_algorithm_round_trip(name: &str, algorithm: &Algorithm) {
    let (features, targets) = regression_testing_data();
    let settings = Settings::default().with_number_of_folds(4).only(algorithm);
    let mut model = Model::new(features, targets, settings);
    model.train().unwrap();

    let expected = model.predict(inference_rows()).unwrap();
    let artifact = TempModelFile::new(name);
    model.save(&artifact.0).unwrap();
    let loaded = Model::load(&artifact.0).unwrap();
    let actual = loaded.predict(inference_rows()).unwrap();
    assert_predictions_match(&expected, &actual);
    let resaved = TempModelFile::new("resaved-family");
    loaded.save(&resaved.0).unwrap();
    let reloaded = Model::load(&resaved.0).unwrap();
    assert_predictions_match(&expected, &reloaded.predict(inference_rows()).unwrap());
    assert_invalid_fitted_state_rejected(name, &artifact.0);
}

fn small_data() -> (DenseMatrix<f64>, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let x = f64::from(i) / 24.0;
            vec![x, x * x + 0.1, f64::from(i % 3)]
        })
        .collect();
    let targets = rows.iter().map(|r| 2.0 * r[0] - r[1] + r[2]).collect();
    (DenseMatrix::from_2d_vec(&rows).unwrap(), targets)
}

fn round_trip_settings(settings: Settings) {
    let (features, targets) = small_data();
    let mut model = Model::new(features.clone(), targets, settings.with_number_of_folds(3));
    model.train().unwrap();
    let expected = model.predict(features.clone()).unwrap();
    let artifact = TempModelFile::new("configured-roundtrip");
    model.save(&artifact.0).unwrap();
    let loaded = Model::load(&artifact.0).unwrap();
    assert_predictions_match(&expected, &loaded.predict(features.clone()).unwrap());
    loaded.save(&artifact.0).unwrap();
    assert_predictions_match(
        &expected,
        &Model::load(&artifact.0).unwrap().predict(features).unwrap(),
    );
}

#[test]
fn every_svr_kernel_round_trips() {
    for kernel in [
        Kernel::Linear,
        Kernel::RBF(0.25),
        Kernel::Polynomial(2.0, 0.25, 1.0),
        Kernel::Sigmoid(0.1, 0.0),
    ] {
        round_trip_settings(
            Settings::default()
                .with_svr_settings(SVRParameters::default().with_kernel(kernel))
                .only(&Algorithm::default_support_vector_regressor()),
        );
    }
}

#[test]
fn knn_distances_and_search_algorithms_round_trip() {
    use automl::settings::KNNAlgorithmName;
    use automl::utils::distance::Distance;
    for distance in [
        Distance::Euclidean,
        Distance::Manhattan,
        Distance::Minkowski(3),
        Distance::Hamming,
    ] {
        for search in [KNNAlgorithmName::LinearSearch, KNNAlgorithmName::CoverTree] {
            round_trip_settings(
                Settings::default()
                    .with_knn_regressor_settings(
                        KNNParameters::default()
                            .with_distance(distance)
                            .with_algorithm(search),
                    )
                    .only(&Algorithm::default_knn_regressor()),
            );
        }
    }
}

#[test]
fn preprocessing_variants_round_trip() {
    use automl::settings::{
        CategoricalEncoderParams, CategoricalEncoding, ColumnFilterParams, PowerTransformParams,
    };
    for step in [
        PreprocessingStep::AddInteractions,
        PreprocessingStep::AddPolynomial { order: 2 },
        PreprocessingStep::ReplaceWithPCA {
            number_of_components: 2,
        },
        PreprocessingStep::ReplaceWithSVD {
            number_of_components: 2,
        },
        PreprocessingStep::Standardize(StandardizeParams::default()),
        PreprocessingStep::Scale(ScaleParams::default()),
        PreprocessingStep::Impute(ImputeParams::default()),
        PreprocessingStep::EncodeCategorical(CategoricalEncoderParams::default()),
        PreprocessingStep::EncodeCategorical(CategoricalEncoderParams {
            encoding: CategoricalEncoding::one_hot(false),
            selector: ColumnSelector::Include(vec![2]),
        }),
        PreprocessingStep::PowerTransform(PowerTransformParams::default()),
        PreprocessingStep::FilterColumns(ColumnFilterParams::default()),
    ] {
        round_trip_settings(
            Settings::default()
                .with_preprocessing(PreprocessingPipeline::new().add_step(step))
                .only(&Algorithm::default_ridge()),
        );
    }
}

#[test]
fn f32_models_round_trip() {
    type SmallModel = RegressionModel<f32, f32, DenseMatrix<f32>, Vec<f32>>;
    type SmallAlgorithm = RegressionAlgorithm<f32, f32, DenseMatrix<f32>, Vec<f32>>;
    let rows: Vec<Vec<f32>> = (0_u16..16)
        .map(|i| vec![f32::from(i) / 16.0, f32::from(i % 3)])
        .collect();
    let targets: Vec<f32> = rows.iter().map(|r| r[0] + 2.0 * r[1]).collect();
    let features = DenseMatrix::from_2d_vec(&rows).unwrap();
    for algorithm in [
        SmallAlgorithm::default_ridge(),
        SmallAlgorithm::default_support_vector_regressor(),
    ] {
        let settings = RegressionSettings::default()
            .with_number_of_folds(3)
            .only(&algorithm);
        let mut model = SmallModel::new(features.clone(), targets.clone(), settings);
        model.train().unwrap();
        let expected = model.predict(features.clone()).unwrap();
        let artifact = TempModelFile::new("float32");
        model.save(&artifact.0).unwrap();
        let actual = SmallModel::load(&artifact.0)
            .unwrap()
            .predict(features.clone())
            .unwrap();
        for (a, b) in expected.iter().zip(actual) {
            assert!((*a - b).abs() < 1e-5);
        }
    }
}

#[test]
fn saving_after_failed_retraining_preserves_last_predictions() {
    let (features, targets) = small_data();
    let mut model = Model::new(
        features.clone(),
        targets,
        Settings::default()
            .with_number_of_folds(3)
            .only(&Algorithm::default_ridge()),
    );
    model.train().unwrap();
    let expected = model.predict(features.clone()).unwrap();
    model.settings = Settings::default()
        .with_number_of_folds(1)
        .only(&Algorithm::default_linear());
    assert!(model.train().is_err());
    let artifact = TempModelFile::new("failed-retrain");
    model.save(&artifact.0).unwrap();
    let mut loaded = Model::load(&artifact.0).unwrap();
    assert_predictions_match(&expected, &loaded.predict(features.clone()).unwrap());
    assert!(loaded.train().is_err());
    loaded.save(&artifact.0).unwrap();
    assert_predictions_match(
        &expected,
        &Model::load(&artifact.0).unwrap().predict(features).unwrap(),
    );
}

#[test]
fn constant_svr_without_support_vectors_round_trips() {
    let (features, _) = small_data();
    let settings = Settings::default()
        .with_number_of_folds(3)
        .with_svr_settings(SVRParameters::default().with_eps(100.0))
        .only(&Algorithm::default_support_vector_regressor());
    let mut model = Model::new(features.clone(), vec![3.0; 24], settings);
    model.train().unwrap();
    let expected = model.predict(features.clone()).unwrap();
    let artifact = TempModelFile::new("constant-svr");
    model.save(&artifact.0).unwrap();
    let loaded = Model::load(&artifact.0).unwrap();
    assert_predictions_match(&expected, &loaded.predict(features).unwrap());
    assert!(
        loaded
            .predict(DenseMatrix::from_2d_array(&[&[1.0]]).unwrap())
            .is_err()
    );
}

fn assert_invalid_fitted_state_rejected(name: &str, artifact: &Path) {
    let mut encoded: serde_json::Value =
        serde_json::from_slice(&fs::read(artifact).unwrap()).unwrap();
    let algorithm = encoded["algorithm"]["algorithm"].as_str().unwrap();
    match algorithm {
        "linear" | "ridge" | "lasso" | "elastic_net" => {
            encoded["algorithm"]["model"]["coefficients"] = serde_json::Value::Null;
        }
        "decision_tree_regressor" => {
            encoded["algorithm"]["model"]["tree_regressor"] = serde_json::Value::Null;
        }
        "random_forest_regressor" | "extra_trees_regressor" => {
            encoded["algorithm"]["model"]["forest_regressor"] = serde_json::Value::Null;
        }
        "k_n_n_regressor" => {
            encoded["algorithm"]["model"]["y"] = serde_json::Value::Null;
        }
        "support_vector_regressor" => {
            encoded["algorithm"]["model"] = serde_json::Value::Null;
        }
        other => panic!("unexpected persisted algorithm {other}"),
    }
    fs::write(artifact, serde_json::to_vec_pretty(&encoded).unwrap()).unwrap();

    let error = Model::load(artifact)
        .err()
        .expect("invalid fitted state should be rejected during load");
    assert!(
        matches!(error, PersistenceError::InvalidFormat(_)),
        "unexpected error for {name}: {error}"
    );
}

#[test]
fn supported_regressors_round_trip_through_files() {
    for (name, algorithm) in [
        ("linear", Algorithm::default_linear()),
        ("ridge", Algorithm::default_ridge()),
        ("lasso", Algorithm::default_lasso()),
        ("elastic-net", Algorithm::default_elastic_net()),
        ("random-forest", Algorithm::default_random_forest()),
        ("extra-trees", Algorithm::default_extra_trees_regressor()),
        ("decision-tree", Algorithm::default_decision_tree()),
        ("knn", Algorithm::default_knn_regressor()),
    ] {
        assert_algorithm_round_trip(name, &algorithm);
    }
}

#[test]
fn save_atomically_replaces_an_existing_artifact() {
    let (features, targets) = regression_testing_data();
    let settings = Settings::default()
        .with_number_of_folds(4)
        .only(&Algorithm::default_ridge());
    let mut first = Model::new(features.clone(), targets.clone(), settings);
    first.train().unwrap();

    let artifact = TempModelFile::new("atomic-overwrite");
    first.save(&artifact.0).unwrap();
    let first_bytes = fs::read(&artifact.0).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        assert_eq!(
            fs::metadata(&artifact.0).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::set_permissions(&artifact.0, fs::Permissions::from_mode(0o640)).unwrap();
    }

    let shifted_targets = targets.into_iter().map(|target| target + 500.0).collect();
    let settings = Settings::default()
        .with_number_of_folds(4)
        .only(&Algorithm::default_ridge());
    let mut replacement = Model::new(features, shifted_targets, settings);
    replacement.train().unwrap();
    let expected = replacement.predict(inference_rows()).unwrap();
    replacement.save(&artifact.0).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        assert_eq!(
            fs::metadata(&artifact.0).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    assert_ne!(fs::read(&artifact.0).unwrap(), first_bytes);
    let loaded = Model::load(&artifact.0).unwrap();
    assert_predictions_match(&expected, &loaded.predict(inference_rows()).unwrap());
}

#[test]
fn knn_artifact_retains_training_examples_needed_for_inference() {
    const DISTINCTIVE_FEATURE: f64 = 91_234.567_89;
    const DISTINCTIVE_TARGET: f64 = -76_543.210_98;

    let features = DenseMatrix::from_2d_array(&[
        &[DISTINCTIVE_FEATURE, 1.0],
        &[2.0, 2.0],
        &[3.0, 3.0],
        &[4.0, 4.0],
    ])
    .unwrap();
    let targets = vec![DISTINCTIVE_TARGET, 20.0, 30.0, 40.0];
    let settings = Settings::default()
        .with_number_of_folds(2)
        .with_knn_regressor_settings(KNNParameters::default().with_k(1))
        .only(&Algorithm::default_knn_regressor());
    let mut model = Model::new(features, targets, settings);
    model.train().unwrap();

    let inference =
        DenseMatrix::from_2d_array(&[&[DISTINCTIVE_FEATURE, 1.0], &[3.0, 3.0]]).unwrap();
    let expected = model.predict(inference.clone()).unwrap();
    let artifact = TempModelFile::new("knn-sensitive-state");
    model.save(&artifact.0).unwrap();

    let encoded: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifact.0).unwrap()).unwrap();
    assert!(
        json_contains_number(&encoded["algorithm"], DISTINCTIVE_FEATURE),
        "KNN artifact should retain a distinctive training feature"
    );
    assert!(
        json_contains_number(&encoded["algorithm"], DISTINCTIVE_TARGET),
        "KNN artifact should retain a distinctive training target"
    );

    let loaded = Model::load(&artifact.0).unwrap();
    let actual = loaded.predict(inference).unwrap();
    assert_predictions_match(&expected, &actual);
}

#[test]
fn svr_and_fitted_preprocessing_round_trip_together() {
    let (features, targets) = regression_testing_data();
    let pipeline = PreprocessingPipeline::new()
        .add_step(PreprocessingStep::Impute(ImputeParams {
            strategy: ImputeStrategy::Median,
            selector: ColumnSelector::All,
        }))
        .add_step(PreprocessingStep::Scale(ScaleParams {
            selector: ColumnSelector::All,
            strategy: ScaleStrategy::Standard(StandardizeParams::default()),
        }))
        .add_step(PreprocessingStep::ReplaceWithPCA {
            number_of_components: 3,
        });
    let settings = Settings::default()
        .with_number_of_folds(4)
        .with_preprocessing(pipeline)
        .with_svr_settings(
            SVRParameters::default()
                .with_c(1.2)
                .with_eps(0.2)
                .with_kernel(Kernel::RBF(0.25)),
        )
        .only(&Algorithm::default_support_vector_regressor());
    let mut model = Model::new(features, targets, settings);
    model.train().unwrap();

    let inference = DenseMatrix::from_2d_array(&[
        &[f64::NAN, 235.6, 159.0, 107.608, 1947.0, 60.323],
        &[259.426, 232.5, 145.6, 108.632, 1948.0, 61.122],
    ])
    .unwrap();
    let expected = model.predict(inference.clone()).unwrap();
    let artifact = TempModelFile::new("svr-preprocessing");
    model.save(&artifact.0).unwrap();

    let mut loaded = Model::load(&artifact.0).unwrap();
    let actual = loaded.predict(inference.clone()).unwrap();
    assert_predictions_match(&expected, &actual);

    let resaved_artifact = TempModelFile::new("svr-preprocessing-resaved");
    loaded.save(&resaved_artifact.0).unwrap();
    let reloaded = Model::load(&resaved_artifact.0).unwrap();
    let reloaded_predictions = reloaded.predict(inference).unwrap();
    assert_predictions_match(&expected, &reloaded_predictions);
    assert_invalid_fitted_state_rejected("svr", &resaved_artifact.0);

    let mut encoded: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifact.0).unwrap()).unwrap();
    assert_eq!(encoded["format"], "automl-supervised-model");
    assert_eq!(encoded["version"], 1);
    assert_eq!(encoded["smartcore_version"], "0.4.2");
    assert!(encoded.get("x_train_raw").is_none());
    assert!(encoded.get("comparison").is_none());

    encoded["preprocessor"]["trained_steps"][0]["Impute"]["values"] =
        serde_json::Value::Array(Vec::new());
    let invalid_preprocessor = TempModelFile::new("invalid-preprocessing");
    fs::write(
        &invalid_preprocessor.0,
        serde_json::to_vec_pretty(&encoded).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        Model::load(&invalid_preprocessor.0),
        Err(PersistenceError::InvalidFormat(_))
    ));

    let error = loaded.train().unwrap_err();
    assert!(error.to_string().contains("inference-only"));
}

#[test]
fn save_rejects_untrained_and_xgboost_models_explicitly() {
    let (features, targets) = regression_testing_data();
    let artifact = TempModelFile::new("untrained");
    let untrained = Model::new(features.clone(), targets.clone(), Settings::default());
    assert_eq!(
        untrained.save(&artifact.0),
        Err(PersistenceError::NotTrained)
    );

    let good_settings = Settings::default()
        .with_number_of_folds(4)
        .only(&Algorithm::default_ridge());
    let mut good_model = Model::new(features.clone(), targets.clone(), good_settings);
    good_model.train().unwrap();
    good_model.save(&artifact.0).unwrap();
    let good_artifact = fs::read(&artifact.0).unwrap();

    let settings = Settings::default()
        .with_number_of_folds(4)
        .with_xgboost_settings(XGRegressorParameters::default().with_n_estimators(2))
        .only(&Algorithm::default_xgboost_regressor());
    let mut xgboost = Model::new(features, targets, settings);
    xgboost.train().unwrap();
    let error = xgboost.save(&artifact.0).unwrap_err();
    assert!(matches!(error, PersistenceError::UnsupportedAlgorithm(_)));
    assert!(error.to_string().contains("XGBoost"));
    assert_eq!(fs::read(&artifact.0).unwrap(), good_artifact);
    Model::load(&artifact.0).unwrap();
}

#[test]
fn load_validates_artifact_kind_and_version_before_model_data() {
    let wrong_kind = TempModelFile::new("wrong-kind");
    fs::write(
        &wrong_kind.0,
        r#"{"format":"other","version":1,"smartcore_version":"0.4.2"}"#,
    )
    .unwrap();
    assert!(matches!(
        Model::load(&wrong_kind.0),
        Err(PersistenceError::InvalidFormat(_))
    ));

    let future_version = TempModelFile::new("future-version");
    fs::write(
        &future_version.0,
        r#"{"format":"automl-supervised-model","version":999,"smartcore_version":"0.4.2"}"#,
    )
    .unwrap();
    let error = Model::load(&future_version.0)
        .err()
        .expect("future model version should fail");
    assert_eq!(
        error,
        PersistenceError::UnsupportedVersion {
            supported: 1,
            found: 999,
        }
    );

    let wrong_smartcore = TempModelFile::new("wrong-smartcore");
    fs::write(
        &wrong_smartcore.0,
        r#"{"format":"automl-supervised-model","version":1,"smartcore_version":"0.4.3"}"#,
    )
    .unwrap();
    assert!(matches!(
        Model::load(&wrong_smartcore.0),
        Err(PersistenceError::InvalidFormat(_))
    ));
}
