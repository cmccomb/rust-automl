use smartcore::linalg::basic::matrix::DenseMatrix;

/// Return clustering data for tests and examples.
///
/// # Returns
///
/// * `x` - Feature matrix containing points for clustering.
pub fn clustering_testing_data() -> DenseMatrix<f64> {
    DenseMatrix::from_2d_vec(&vec![
        vec![1.0_f64, 1.0],
        vec![1.2, 0.8],
        vec![8.0, 8.0],
        vec![8.2, 8.2],
    ])
    .unwrap()
}
