use std::error::Error;
use std::path::Path;

use csv::ReaderBuilder;
use smartcore::linalg::basic::matrix::DenseMatrix;

type CsvRows = (Vec<Vec<f64>>, Vec<String>);
type CsvResult = Result<CsvRows, Box<dyn Error>>;

fn load_feature_rows<P: AsRef<Path>>(path: P) -> CsvResult {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for record in reader.records() {
        let record = record?;
        let record_len = record.len();
        if record_len < 2 {
            return Err("dataset requires at least one feature and a target column".into());
        }
        let feature_len = record_len - 1;
        let mut row = Vec::with_capacity(feature_len);
        for value in record.iter().take(feature_len) {
            row.push(value.parse()?);
        }
        let target_value = record
            .get(feature_len)
            .ok_or("dataset missing target column")?;
        features.push(row);
        targets.push(target_value.to_string());
    }

    Ok((features, targets))
}

fn parse_label(raw: &str) -> Result<u32, Box<dyn Error>> {
    let numeric: f64 = raw.parse()?;
    if (numeric - 1.0).abs() < f64::EPSILON {
        Ok(1)
    } else if numeric.abs() < f64::EPSILON {
        Ok(0)
    } else {
        Err("unexpected label".into())
    }
}

/// Load the Wisconsin Diagnostic Breast Cancer dataset from `data/breast_cancer.csv`.
///
/// # Errors
///
/// Returns an error if the CSV file cannot be read or parsed into numeric data.
pub fn load_breast_cancer_dataset() -> Result<(DenseMatrix<f64>, Vec<u32>), Box<dyn Error>> {
    let (feature_rows, raw_targets) = load_feature_rows("data/breast_cancer.csv")?;
    let features = DenseMatrix::from_2d_vec(&feature_rows)?;
    let targets = raw_targets
        .into_iter()
        .map(|value| parse_label(&value))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok((features, targets))
}
