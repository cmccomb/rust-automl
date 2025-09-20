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

/// Return binary feature data for Bernoulli naive Bayes tests.
///
/// # Returns
///
/// * `(x, y)` - Feature matrix with binary entries and corresponding targets.
#[cfg(test)]
pub fn bernoulli_binary_classification_data() -> (DenseMatrix<f64>, Vec<u32>) {
    let x = DenseMatrix::from_2d_array(&[
        &[1.0, 0.0, 1.0],
        &[0.0, 1.0, 0.0],
        &[1.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0],
    ])
    .unwrap();

    let y = vec![1, 0, 1, 0];

    (x, y)
}

/// Return continuous feature data that can be binarized for Bernoulli naive Bayes tests.
///
/// # Returns
///
/// * `(x, y)` - Feature matrix and corresponding targets.
#[cfg(test)]
pub fn bernoulli_threshold_classification_data() -> (DenseMatrix<f64>, Vec<u32>) {
    let x =
        DenseMatrix::from_2d_array(&[&[0.2, 0.8], &[0.7, 0.3], &[0.9, 0.1], &[0.1, 0.9]]).unwrap();

    let y = vec![1, 0, 0, 1];

    (x, y)
}
