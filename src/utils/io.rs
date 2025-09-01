//! I/O utilities.

use csv::ReaderBuilder;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::error::Error;
use std::fs::File;
use std::path::Path;

/// Load a CSV file and return its feature matrix.
///
/// # Arguments
///
/// * `path` - Path to the CSV file.
///
/// # Errors
///
/// Returns an error if the file cannot be read, a value fails to parse into
/// `f64`, or the rows have inconsistent lengths.
///
/// # Examples
///
/// ```
/// use automl::{ClusteringModel, settings::ClusteringSettings};
/// use automl::utils::load_csv_features;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = load_csv_features("tests/fixtures/clustering_points.csv")?;
/// let settings = ClusteringSettings::default().with_k(2);
/// let mut model = ClusteringModel::new(x.clone(), settings);
/// model.train();
/// let clusters: Vec<u8> = model.predict(&x);
/// # Ok(())
/// # }
/// ```
pub fn load_csv_features<P: AsRef<Path>>(path: P) -> Result<DenseMatrix<f64>, Box<dyn Error>> {
    let file = File::open(path.as_ref())?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut features: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row = record
            .iter()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        features.push(row);
    }

    let matrix = DenseMatrix::from_2d_vec(&features)?;
    Ok(matrix)
}

/// Load a labeled CSV file and split features from the target column.
///
/// # Arguments
///
/// * `path` - Path to the CSV file.
/// * `target_col` - Zero-based index of the target column.
///
/// # Errors
///
/// Returns an error if the file cannot be read, a value fails to parse into
/// `f64`, or the rows have inconsistent lengths.
///
/// # Examples
///
/// ```
/// use automl::{RegressionModel, RegressionSettings};
/// use automl::utils::load_labeled_csv;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (x, y) = load_labeled_csv("tests/fixtures/supervised_sample.csv", 2)?;
/// let mut model = RegressionModel::new(x, y, RegressionSettings::default());
/// # Ok(())
/// # }
/// ```
pub fn load_labeled_csv<P: AsRef<Path>>(
    path: P,
    target_col: usize,
) -> Result<(DenseMatrix<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path.as_ref())?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row: Vec<f64> = Vec::new();
        for (idx, field) in record.iter().enumerate() {
            let value: f64 = field.parse()?;
            if idx == target_col {
                targets.push(value);
            } else {
                row.push(value);
            }
        }
        features.push(row);
    }

    let matrix = DenseMatrix::from_2d_vec(&features)?;
    Ok((matrix, targets))
}
