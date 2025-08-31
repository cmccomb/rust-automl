//! Supervised model definition and public API.

/// Feature engineering helpers.
mod feature_engineering;
/// Preprocessing utilities.
mod preprocessing;
/// Model training routines.
mod training;

use crate::settings::{Algorithm, FinalAlgorithm, Settings};
use comfy_table::{
    Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
};
use humantime::format_duration;
use smartcore::decomposition::{pca::PCA, svd::SVD};
use smartcore::linalg::basic::arrays::{Array, Array1, Array2, MutArrayView1};
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::model_selection::CrossValidationResult;
use smartcore::numbers::{floatnum::FloatNumber, realnum::RealNumber};
use std::{
    fmt::{Display, Formatter},
    time::Duration,
};

/// Trains and compares supervised models
pub struct SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Settings for the model.
    settings: Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    /// The training data.
    x_train: InputArray,
    /// The training labels.
    y_train: OutputArray,
    /// The validation data.
    x_val: InputArray,
    /// The validation labels.
    y_val: OutputArray,
    /// The results of the model comparison.
    comparison: Vec<(
        CrossValidationResult,
        Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
        Duration,
    )>,
    /// The final model.
    metamodel: (CrossValidationResult, FinalAlgorithm, Duration),
    /// PCA model for preprocessing.
    preprocessing_pca: Option<PCA<INPUT, InputArray>>,
    /// SVD model for preprocessing.
    preprocessing_svd: Option<SVD<INPUT, InputArray>>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Predict values using the final model based on a matrix.
    ///
    /// ```
    /// # use smartcore::linalg::basic::matrix::DenseMatrix;
    /// # use automl::{regression_testing_data, Settings};
    /// # use automl::supervised_model::SupervisedModel;
    /// # let (x, y) = regression_testing_data();
    /// # let mut model = SupervisedModel::new(
    /// #    x, y,
    /// #    Settings::default_regression()
    /// #        .only(automl::settings::Algorithm::default_linear())
    /// # );
    /// # model.train();
    /// let X = DenseMatrix::from_2d_vec(&vec![vec![5.0; 6]; 5]).unwrap();
    /// model.predict(X);
    /// ```
    ///
    /// # Panics
    ///
    /// If the model has not been trained, this function will panic.
    pub fn predict(&self, x: InputArray) -> OutputArray {
        let x = self.preprocess(x);
        match self.settings.final_model_approach {
            FinalAlgorithm::None => {
                panic!("No final model available. Did you forget to call `train`?")
            }
            FinalAlgorithm::Best => match self.comparison.get(0) {
                Some((_, algorithm, _)) => match algorithm {
                    Algorithm::Linear(model) => model.predict(&x),
                    Algorithm::Lasso(model) => model.predict(&x),
                    Algorithm::Ridge(model) => model.predict(&x),
                    Algorithm::ElasticNet(model) => model.predict(&x),
                    Algorithm::RandomForestRegressor(model) => model.predict(&x),
                    Algorithm::DecisionTreeRegressor(model) => model.predict(&x),
                }
                .expect(
                    "Error during inference. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
                ),
                None => panic!("Model comparison is empty. Did you forget to call `train`?"),
            },
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Build a new supervised model
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `y` - The output data
    /// * `settings` - The settings for the model
    pub fn new(
        x: InputArray,
        y: OutputArray,
        settings: Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        Self {
            settings,
            x_train: x.clone(),
            y_train: y.clone(),
            x_val: x,
            y_val: y,
            comparison: vec![],
            metamodel: (
                CrossValidationResult {
                    test_score: vec![],
                    train_score: vec![],
                },
                FinalAlgorithm::Best,
                Duration::default(),
            ),
            preprocessing_pca: None,
            preprocessing_svd: None,
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber,
    InputArray: SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + Clone
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec![
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new("Time").add_attribute(Attribute::Bold),
            Cell::new(format!("Training {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
        ]);
        for model in &self.comparison {
            let mut row_vec = vec![];
            row_vec.push(format!("{}", &model.1.to_string()));
            row_vec.push(format!("{}", format_duration(model.2)));
            let decider = ((model.0.mean_train_score() + model.0.mean_test_score()) / 2.0).abs();
            if decider > 0.01 && decider < 1000.0 {
                row_vec.push(format!("{:.2}", &model.0.mean_train_score()));
                row_vec.push(format!("{:.2}", &model.0.mean_test_score()));
            } else {
                row_vec.push(format!("{:.3e}", &model.0.mean_train_score()));
                row_vec.push(format!("{:.3e}", &model.0.mean_test_score()));
            }

            table.add_row(row_vec);
        }

        let mut meta_table = Table::new();
        meta_table.load_preset(UTF8_FULL);
        meta_table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        meta_table.set_header(vec![
            Cell::new("Meta Model").add_attribute(Attribute::Bold),
            Cell::new(format!("Training {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
        ]);

        // Populate row
        let mut row_vec = vec![];
        row_vec.push(format!("METAMODEL"));
        let decider = ((self.metamodel.0.mean_train_score() + self.metamodel.0.mean_test_score())
            / 2.0)
            .abs();
        if decider > 0.01 && decider < 1000.0 {
            row_vec.push(format!("{:.2}", self.metamodel.0.mean_train_score()));
            row_vec.push(format!("{:.2}", self.metamodel.0.mean_test_score()));
        } else {
            row_vec.push(format!("{:.3e}", self.metamodel.0.mean_train_score()));
            row_vec.push(format!("{:.3e}", self.metamodel.0.mean_test_score()));
        }

        // Add row to table
        meta_table.add_row(row_vec);

        // Write
        write!(f, "{table}\n{meta_table}")
    }
}

/// This is a wrapper for the `CrossValidationResult`
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "CrossValidationResult")]
struct CrossValidationResultDef {
    /// Vector with test scores on each cv split
    pub test_score: Vec<f64>,
    /// Vector with training scores on each cv split
    pub train_score: Vec<f64>,
}
