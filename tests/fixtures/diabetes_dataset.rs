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

/// Load the diabetes progression dataset from `data/diabetes.csv`.
///
/// # Errors
///
/// Returns an error if the CSV file cannot be read or parsed into numeric data.
pub fn load_diabetes_dataset() -> Result<(DenseMatrix<f64>, Vec<f64>), Box<dyn Error>> {
    let (feature_rows, raw_targets) = load_feature_rows("data/diabetes.csv")?;
    let features = DenseMatrix::from_2d_vec(&feature_rows)?;
    let targets = raw_targets
        .into_iter()
        .map(|value| -> Result<f64, Box<dyn Error>> { Ok(value.parse()?) })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok((features, targets))
}
