//! I/O utilities.

use csv::ReaderBuilder;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::error::Error;
use std::fs::File;
use std::path::Path;

/// Load a labeled CSV file and split features from the target column.
///
/// # Arguments
///
/// * `path` - Path to the CSV file.
/// * `target_col` - Zero-based index of the target column.
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
