#[path = "fixtures/clustering_data.rs"]
mod clustering_data;

use automl::{
    ClusteringModel, ModelError,
    metrics::ClusterMetrics,
    settings::{ClusteringAlgorithmName, ClusteringSettings},
};
use clustering_data::clustering_testing_data;
use smartcore::linalg::basic::matrix::DenseMatrix;

#[test]
fn default_clustering_trains_all_algorithms() {
    let x = clustering_testing_data();
    let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    let labels: Vec<u8> = model.predict(&x).unwrap();
    assert_eq!(labels.len(), 4);

    let algorithms = model.trained_algorithm_names();
    assert_eq!(
        algorithms,
        vec![
            ClusteringAlgorithmName::KMeans,
            ClusteringAlgorithmName::Agglomerative,
            ClusteringAlgorithmName::DBSCAN,
        ]
    );

    for algorithm in algorithms {
        let labels: Vec<u8> = model.predict_with(algorithm, &x).unwrap();
        assert_eq!(labels.len(), 4);
    }
}

#[test]
fn agglomerative_clustering_runs() {
    let x = clustering_testing_data();
    let settings = ClusteringSettings::default()
        .with_k(2)
        .with_algorithm(ClusteringAlgorithmName::Agglomerative);
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();
    let labels: Vec<u8> = model.predict(&x).unwrap();
    assert_eq!(labels.len(), 4);
}

#[test]
fn dbscan_clustering_runs() {
    let x = clustering_testing_data();
    let settings = ClusteringSettings::default()
        .with_algorithm(ClusteringAlgorithmName::DBSCAN)
        .with_min_samples(2);
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();
    let labels: Vec<u8> = model.predict(&x).unwrap();
    assert_eq!(labels.len(), 4);
}

#[test]
fn clustering_metrics_compute() {
    let x = clustering_testing_data();
    let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    let predicted: Vec<u8> = model.predict(&x).unwrap();
    let truth = vec![1_u8, 1, 2, 2];
    let mut scores = ClusterMetrics::<u8>::hcv_score();
    scores.compute(&truth, &predicted);
    assert!((scores.homogeneity().unwrap() - 1.0).abs() < 1e-12);
    assert!((scores.completeness().unwrap() - 1.0).abs() < 1e-12);
    assert!((scores.v_measure().unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn clustering_model_display_shows_metrics() {
    // Arrange
    let x = clustering_testing_data();
    let mut model: ClusteringModel<f64, u8, DenseMatrix<f64>, Vec<u8>> =
        ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    let truth = vec![1_u8, 1, 2, 2];
    model.evaluate(&truth);

    // Act
    let output = format!("{model}");

    // Assert
    assert!(output.contains("KMeans"));
    assert!(output.contains("Agglomerative"));
    assert!(output.contains("DBSCAN"));
    assert!(output.contains("V-Measure"));
    assert!(output.contains("1.00"));
}

#[test]
fn clustering_model_display_shows_configured_algorithm_when_untrained() {
    // Arrange
    let x = clustering_testing_data();
    let settings = ClusteringSettings::default().with_algorithm(ClusteringAlgorithmName::DBSCAN);
    let model: ClusteringModel<f64, u8, DenseMatrix<f64>, Vec<u8>> =
        ClusteringModel::new(x, settings);

    // Act
    let output = format!("{model}");

    // Assert
    assert!(output.contains("DBSCAN (untrained)"));
    assert!(output.contains("Homogeneity"));
}

#[test]
fn clustering_model_display_clears_metrics_after_retraining() {
    // Arrange
    let x = clustering_testing_data();
    let truth = vec![1_u8, 1, 2, 2];
    let mut model: ClusteringModel<f64, u8, DenseMatrix<f64>, Vec<u8>> =
        ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    model.evaluate(&truth);

    // Sanity check – metrics should be displayed after evaluation
    let evaluated_output = format!("{model}");
    assert!(evaluated_output.contains("1.00"));

    // Act
    model.train();
    let retrained_output = format!("{model}");

    // Assert
    assert!(!retrained_output.contains("1.00"));
}

#[test]
fn predict_without_training_returns_error() {
    let x = clustering_testing_data();
    let model: ClusteringModel<f64, u8, automl::DenseMatrix<f64>, Vec<u8>> =
        ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, ModelError::NotTrained));
}
