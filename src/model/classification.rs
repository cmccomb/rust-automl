//! Implementation of classification model training and evaluation.

use super::{comparison::ComparisonEntry, preprocessing::Preprocessor};
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
    /// Predict values using the final model based on a vec.
    /// # Panics
    /// If the model has not been trained, this function will panic.
    pub fn predict(self, x: InputArray) -> OutputArray {
        let x = self
            .preprocessor
            .preprocess(x, &self.settings.preprocessing)
            .expect("Cannot preprocess features");
        match self.settings.final_model_approach {
            FinalAlgorithm::None => panic!(""),
            FinalAlgorithm::Best => match &self
                .comparison
                .first()
                .expect("")
                .algorithm
            {
                ClassificationAlgorithm::DecisionTreeClassifier(model) => model.predict(&x),
                ClassificationAlgorithm::KNNClassifier(model) => model.predict(&x),
                ClassificationAlgorithm::RandomForestClassifier(model) => model.predict(&x),
                ClassificationAlgorithm::LogisticRegression(model) => model.predict(&x),
            }
            .expect(
                "Error during inference. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
            ),
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
