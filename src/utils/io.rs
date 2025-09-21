//! I/O utilities.

use csv::{ReaderBuilder, StringRecord};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::num::ParseFloatError;
use std::path::Path;

/// Errors that can occur when loading CSV data.
#[derive(Debug)]
pub enum CsvError {
    /// Failure in underlying I/O operations.
    Io(std::io::Error),
    /// Failure while parsing numeric values or CSV records.
    Parse(Box<dyn Error + Send + Sync>),
    /// Mismatched row lengths when constructing the feature matrix.
    Shape(String),
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsvError::Io(e) => write!(f, "I/O error: {e}"),
            CsvError::Parse(e) => write!(f, "Parse error: {e}"),
            CsvError::Shape(e) => write!(f, "Shape error: {e}"),
        }
    }
}

impl Error for CsvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CsvError::Io(e) => Some(e),
            CsvError::Parse(e) => Some(&**e),
            CsvError::Shape(_) => None,
        }
    }
}

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
/// Row numbers mentioned in error messages are one-based and refer to data rows,
/// excluding the header.
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
/// let clusters: Vec<u8> = model.predict(&x)?;
/// # Ok(())
/// # }
/// ```
pub fn load_csv_features<P: AsRef<Path>>(path: P) -> Result<DenseMatrix<f64>, CsvError> {
    let mut reader = build_csv_reader(path.as_ref())?;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut expected_width: Option<usize> = None;

    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| CsvError::Parse(Box::new(e)))?;
        let row = parse_feature_row(&record, row_idx)?;
        ensure_consistent_width(&row, row_idx, &mut expected_width)?;
        features.push(row);
    }

    if features.is_empty() {
        return Err(CsvError::Shape("no rows found".to_string()));
    }

    let matrix = DenseMatrix::from_2d_vec(&features).map_err(|e| CsvError::Shape(e.to_string()))?;
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
/// Row numbers mentioned in error messages are one-based and refer to data rows,
/// excluding the header.
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
) -> Result<(DenseMatrix<f64>, Vec<f64>), CsvError> {
    let mut reader = build_csv_reader(path.as_ref())?;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();
    let mut expected_width: Option<usize> = None;

    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| CsvError::Parse(Box::new(e)))?;
        let (row, target) = parse_labeled_row(&record, row_idx, target_col)?;
        ensure_consistent_width(&row, row_idx, &mut expected_width)?;
        targets.push(target);
        features.push(row);
    }

    if features.is_empty() {
        return Err(CsvError::Shape("no rows found".to_string()));
    }

    let matrix = DenseMatrix::from_2d_vec(&features).map_err(|e| CsvError::Shape(e.to_string()))?;
    Ok((matrix, targets))
}

fn build_csv_reader(path: &Path) -> Result<csv::Reader<File>, CsvError> {
    let file = File::open(path).map_err(CsvError::Io)?;
    Ok(ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file))
}

fn parse_feature_row(record: &StringRecord, row_idx: usize) -> Result<Vec<f64>, CsvError> {
    if record.is_empty() {
        return Err(CsvError::Shape(format!(
            "row {}: expected at least one column",
            row_idx + 1
        )));
    }

    record
        .iter()
        .enumerate()
        .map(|(col_idx, value)| parse_numeric_field(value, row_idx, col_idx))
        .collect()
}

fn parse_labeled_row(
    record: &StringRecord,
    row_idx: usize,
    target_col: usize,
) -> Result<(Vec<f64>, f64), CsvError> {
    if record.len() <= target_col {
        return Err(CsvError::Shape(format!(
            "row {}: target column index {} out of bounds (row has {} columns)",
            row_idx + 1,
            target_col,
            record.len()
        )));
    }

    if record.len() <= 1 {
        return Err(CsvError::Shape(format!(
            "row {}: expected at least one feature column in addition to the target",
            row_idx + 1
        )));
    }

    let mut target = None;
    let mut row = Vec::with_capacity(record.len() - 1);

    for (col_idx, value) in record.iter().enumerate() {
        let parsed = parse_numeric_field(value, row_idx, col_idx)?;
        if col_idx == target_col {
            target = Some(parsed);
        } else {
            row.push(parsed);
        }
    }

    match target {
        Some(target_value) => Ok((row, target_value)),
        None => Err(CsvError::Shape(format!(
            "row {}: missing target column {}",
            row_idx + 1,
            target_col
        ))),
    }
}

fn parse_numeric_field(value: &str, row_idx: usize, col_idx: usize) -> Result<f64, CsvError> {
    value.parse::<f64>().map_err(|err: ParseFloatError| {
        CsvError::Parse(Box::new(FloatParseError::new(
            row_idx + 1,
            col_idx + 1,
            err,
        )))
    })
}

fn ensure_consistent_width(
    row: &[f64],
    row_idx: usize,
    expected_width: &mut Option<usize>,
) -> Result<(), CsvError> {
    if row.is_empty() {
        return Err(CsvError::Shape(format!(
            "row {}: expected at least one column",
            row_idx + 1
        )));
    }

    match expected_width {
        Some(width) if row.len() != *width => Err(CsvError::Shape(format!(
            "row {}: expected {} columns but found {}",
            row_idx + 1,
            width,
            row.len()
        ))),
        Some(_) => Ok(()),
        None => {
            *expected_width = Some(row.len());
            Ok(())
        }
    }
}

#[derive(Debug)]
struct FloatParseError {
    row: usize,
    column: usize,
    source: ParseFloatError,
}

impl FloatParseError {
    fn new(row: usize, column: usize, source: ParseFloatError) -> Self {
        Self {
            row,
            column,
            source,
        }
    }
}

impl fmt::Display for FloatParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to parse float at row {}, column {}: {}",
            self.row, self.column, self.source
        )
    }
}

impl Error for FloatParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}
