//! Implementation of classification model training and evaluation.

use super::{ModelError, ModelResult, comparison::ComparisonEntry, preprocessing::Preprocessor};
use crate::algorithms::ClassificationAlgorithm;
use crate::settings::{ClassificationSettings, FinalAlgorithm, Metric};
use smartcore::{
    linalg::{
        basic::arrays::{Array, Array1, Array2, MutArrayView1},
        traits::{
            cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
            svd::SVDDecomposable,
        },
    },
    numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber},
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
};
use {
    comfy_table::{
        Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
    },
    humantime::format_duration,
};

/// Trains and compares classification models
pub struct ClassificationModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
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
    settings: ClassificationSettings,
    /// The training data.
    x_train: InputArray,
    /// The training labels.
    y_train: OutputArray,
    /// The results of the model comparison.
    comparison:
        Vec<ComparisonEntry<ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>>>,
    /// Preprocessor responsible for feature engineering.
    preprocessor: Preprocessor<INPUT, InputArray>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    ClassificationModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: Clone
        + Array<INPUT, (usize, usize)>
        + Array2<INPUT>
        + EVDDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Predict values using the final model based on a feature matrix.
    ///
    /// # Examples
    /// ```
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    /// use automl::{ClassificationModel, settings::ClassificationSettings};
    /// let x = DenseMatrix::from_2d_array(&[
    ///     &[0.0, 0.0],
    ///     &[1.0, 1.0],
    ///     &[0.0, 1.0],
    ///     &[1.0, 0.0],
    ///     &[0.5, 0.5],
    ///     &[1.5, 1.5],
    ///     &[0.5, 1.5],
    ///     &[1.5, 0.5],
    /// ]).unwrap();
    /// let y = vec![0_i32, 1, 0, 1, 0, 1, 0, 1];
    /// let mut model = ClassificationModel::new(
    ///     x,
    ///     y,
    ///     ClassificationSettings::default().with_number_of_folds(2),
    /// );
    /// model.train();
    /// let preds = model.predict(
    ///     DenseMatrix::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap(),
    /// );
    /// assert!(preds.is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::NotTrained`] if the model has not been trained or
    /// [`ModelError::Inference`] if the underlying algorithm fails.
    pub fn predict(&self, x: InputArray) -> ModelResult<OutputArray> {
        let x = self
            .preprocessor
            .preprocess(x, &self.settings.preprocessing)
            .map_err(|e| ModelError::Inference(e.to_string()))?;

        match self.settings.final_model_approach {
            FinalAlgorithm::None => Err(ModelError::NotTrained),
            FinalAlgorithm::Best => {
                let entry = self.comparison.first().ok_or(ModelError::NotTrained)?;
                match &entry.algorithm {
                    ClassificationAlgorithm::DecisionTreeClassifier(model) => model.predict(&x),
                    ClassificationAlgorithm::KNNClassifier(model) => model.predict(&x),
                    ClassificationAlgorithm::RandomForestClassifier(model) => model.predict(&x),
                    ClassificationAlgorithm::LogisticRegression(model) => model.predict(&x),
                }
                .map_err(|e| ModelError::Inference(e.to_string()))
            }
        }
    }

    /// Runs a model comparison and trains a final model.
    pub fn train(&mut self) {
        // Train any necessary preprocessing
        self.preprocessor
            .train(&self.x_train.clone(), &self.settings.preprocessing);

        // Iterate over variants in Algorithm
        for alg in ClassificationAlgorithm::all_algorithms(&self.settings) {
            self.record_trained_model(alg.cross_validate_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    ClassificationModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
    InputArray: SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + Clone
        + Array<INPUT, (usize, usize)>
        + CholeskyDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Clone + MutArrayView1<OUTPUT> + Array1<OUTPUT>,
{
    /// Build a new classification model
    pub fn new(x: InputArray, y: OutputArray, settings: ClassificationSettings) -> Self {
        Self {
            settings,
            x_train: x,
            y_train: y,
            comparison: vec![],
            preprocessor: Preprocessor::new(),
        }
    }

    /// Record a model in the comparison.
    fn record_trained_model(
        &mut self,
        trained_model: ComparisonEntry<
            ClassificationAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
        >,
    ) {
        self.comparison.push(trained_model);
        self.sort();
    }

    /// Sort the models in the comparison by their mean test scores.
    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            a.result
                .mean_test_score()
                .partial_cmp(&b.result.mean_test_score())
                .unwrap_or(Equal)
        });
        if matches!(self.settings.sort_by, Metric::RSquared | Metric::Accuracy) {
            self.comparison.reverse();
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for ClassificationModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: Number + Ord,
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
            row_vec.push(model.algorithm.to_string());
            row_vec.push(format_duration(model.duration).to_string());
            let decider = f64::midpoint(
                model.result.mean_train_score(),
                model.result.mean_test_score(),
            )
            .abs();
            if decider > 0.01 && decider < 1000.0 {
                row_vec.push(format!("{:.2}", &model.result.mean_train_score()));
                row_vec.push(format!("{:.2}", &model.result.mean_test_score()));
            } else {
                row_vec.push(format!("{:.3e}", &model.result.mean_train_score()));
                row_vec.push(format!("{:.3e}", &model.result.mean_test_score()));
            }

            table.add_row(row_vec);
        }

        write!(f, "{table}")
    }
}
