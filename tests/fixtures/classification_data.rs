use smartcore::linalg::basic::matrix::DenseMatrix;

/// Return classification data for tests and examples.
///
/// # Returns
///
/// * `(x, y)` - Feature matrix and target vector.
pub fn classification_testing_data() -> (DenseMatrix<f64>, Vec<u32>) {
    let x = DenseMatrix::from_2d_array(&[
        &[0.0, 0.0],
        &[0.1, 0.0],
        &[0.0, 0.1],
        &[0.1, 0.1],
        &[0.9, 0.9],
        &[1.0, 0.9],
        &[0.9, 1.0],
        &[1.0, 1.0],
        &[0.0, 0.2],
        &[0.2, 0.0],
        &[0.0, 0.3],
        &[0.3, 0.0],
        &[0.8, 0.7],
        &[0.7, 0.8],
        &[0.8, 0.8],
        &[0.7, 0.7],
    ])
    .unwrap();

    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];

    (x, y)
}
