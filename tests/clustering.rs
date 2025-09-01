#[path = "fixtures/clustering_data.rs"]
mod clustering_data;

use automl::{
    ClusteringModel,
    metrics::ClusterMetrics,
    settings::{ClusteringAlgorithmName, ClusteringSettings},
};
use clustering_data::clustering_testing_data;

#[test]
fn kmeans_clustering_runs() {
    let x = clustering_testing_data();
    let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    let labels: Vec<u8> = model.predict(&x);
    assert_eq!(labels.len(), 4);
}

#[test]
fn agglomerative_clustering_runs() {
    let x = clustering_testing_data();
    let settings = ClusteringSettings::default()
        .with_k(2)
        .with_algorithm(ClusteringAlgorithmName::Agglomerative);
    let mut model = ClusteringModel::new(x.clone(), settings);
    model.train();
    let labels: Vec<u8> = model.predict(&x);
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
    let labels: Vec<u8> = model.predict(&x);
    assert_eq!(labels.len(), 4);
}

#[test]
fn clustering_metrics_compute() {
    let x = clustering_testing_data();
    let mut model = ClusteringModel::new(x.clone(), ClusteringSettings::default().with_k(2));
    model.train();
    let predicted: Vec<u8> = model.predict(&x);
    let truth = vec![1_u8, 1, 2, 2];
    let mut scores = ClusterMetrics::<u8>::hcv_score();
    scores.compute(&truth, &predicted);
    assert!((scores.homogeneity().unwrap() - 1.0).abs() < 1e-12);
    assert!((scores.completeness().unwrap() - 1.0).abs() < 1e-12);
    assert!((scores.v_measure().unwrap() - 1.0).abs() < 1e-12);
}
