//! Implementation of regression model training and evaluation.

use super::{comparison::ComparisonEntry, preprocessing::Preprocessor};
use crate::algorithms::RegressionAlgorithm;
use crate::settings::{FinalAlgorithm, Metric, RegressionSettings};
use smartcore::{
    linalg::{
        basic::arrays::{Array, Array1, Array2, MutArrayView1},
        traits::{
            cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
            svd::SVDDecomposable,
        },
    },
    model_selection::CrossValidationResult,
    numbers::{floatnum::FloatNumber, realnum::RealNumber},
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    time::Duration,
};
use {
    comfy_table::{
        Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
    },
    humantime::format_duration,
};

/// Trains and compares regression models
pub struct RegressionModel<INPUT, OUTPUT, InputArray, OutputArray>
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
    settings: RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    /// The training data.
    x_train: InputArray,
    /// The training labels.
    y_train: OutputArray,
    /// The results of the model comparison.
    comparison: Vec<ComparisonEntry<RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>>>,
    /// The final model.
    metamodel: (CrossValidationResult, FinalAlgorithm, Duration),
    /// Preprocessor responsible for feature engineering.
    preprocessor: Preprocessor<INPUT, InputArray>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray> RegressionModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber, // + Eq + Hash,
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
    /// ```
    /// # use smartcore::linalg::basic::matrix::DenseMatrix;
    /// # use automl::{algorithms, RegressionModel, RegressionSettings};
    /// # let x = DenseMatrix::from_2d_vec(
    /// #     &(0..12).map(|i| vec![i as f64 + 1.0; 6]).collect::<Vec<_>>(),
    /// # )
    /// # .unwrap();
    /// # let y = vec![0.0_f64; 12];
    /// # let mut model = RegressionModel::new(
    /// #    x,
    /// #    y,
    /// #    RegressionSettings::default().with_number_of_folds(3)
    /// #        .only(&algorithms::RegressionAlgorithm::default_linear()),
    /// # );
    /// # model.train();
    /// let X = DenseMatrix::from_2d_vec(&vec![vec![5.0; 6]; 5]).unwrap();
    /// model.predict(X);
    /// ```
    /// # Panics
    ///
    /// If the model has not been trained, this function will panic.
    pub fn predict(self, x: InputArray) -> OutputArray {
        let x = self
            .preprocessor
            .preprocess(x, &self.settings.preprocessing);
        match self.settings.final_model_approach {
            FinalAlgorithm::None => panic!(""),
            FinalAlgorithm::Best => match &self
                .comparison
                .first()
                .expect("")
                .algorithm
            {
                RegressionAlgorithm::Linear(model) => model.predict(&x),
                RegressionAlgorithm::Lasso(model) => model.predict(&x),
                RegressionAlgorithm::Ridge(model) => model.predict(&x),
                RegressionAlgorithm::ElasticNet(model) => model.predict(&x),
                RegressionAlgorithm::RandomForestRegressor(model) => model.predict(&x),
                RegressionAlgorithm::DecisionTreeRegressor(model) => model.predict(&x),
                RegressionAlgorithm::KNNRegressorHamming(model) => model.predict(&x),
                RegressionAlgorithm::KNNRegressorEuclidian(model) => model.predict(&x),
                RegressionAlgorithm::KNNRegressorManhattan(model) => model.predict(&x),
                RegressionAlgorithm::KNNRegressorMinkowski(model) => model.predict(&x),
            }
            .expect(
                "Error during inference. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
            ),
            // FinalAlgorithm::Blending { .. } => self.predict_by_model(x, top_model),
            // self.predict_blended_model(x, algorithm),
        }
    }

    /// Runs a model comparison and trains a final model.
    /// ```
    /// # use automl::{algorithms, RegressionModel, RegressionSettings};
    /// # use smartcore::linalg::basic::matrix::DenseMatrix;
    /// # let x = DenseMatrix::from_2d_vec(
    /// #     &(0..12).map(|i| vec![i as f64 + 1.0; 6]).collect::<Vec<_>>(),
    /// # )
    /// # .unwrap();
    /// # let y = vec![0.0_f64; 12];
    /// let mut model = RegressionModel::new(
    ///     x,
    ///     y,
    ///     RegressionSettings::default().with_number_of_folds(3)
    /// #        .only(&algorithms::RegressionAlgorithm::default_linear())
    /// );
    /// model.train();
    /// ```
    pub fn train(&mut self) {
        // Train any necessary preprocessing
        self.preprocessor
            .train(&self.x_train.clone(), &self.settings.preprocessing);

        // Iterate over variants in RegressionAlgorithm
        for alg in RegressionAlgorithm::all_algorithms(&self.settings) {
            if !self.settings.skiplist.contains(&alg) {
                self.record_trained_model(alg.cross_validate_model(
                    &self.x_train,
                    &self.y_train,
                    &self.settings,
                ));
            }
        }

        // if let FinalAlgorithm::Blending {
        //     algorithm,
        //     meta_training_fraction,
        //     meta_testing_fraction,
        // } = self.settings.final_model_approach
        // {
        //     self.train_blended_model(algorithm, meta_training_fraction, meta_testing_fraction);
        // }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> RegressionModel<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: RealNumber + FloatNumber,
    OUTPUT: FloatNumber, // + Eq + Hash,
    InputArray: SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + Clone
        + Array<INPUT, (usize, usize)>
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
        settings: RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        Self {
            settings,
            x_train: x,
            // number_of_classes: Self::count_classes(&y),
            y_train: y,
            comparison: vec![],
            metamodel: (
                CrossValidationResult {
                    test_score: vec![],
                    train_score: vec![],
                },
                FinalAlgorithm::Best,
                Duration::default(),
            ),
            preprocessor: Preprocessor::new(),
        }
    }

    /// Record a model in the comparison.
    fn record_trained_model(
        &mut self,
        trained_model: ComparisonEntry<RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>>,
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
        if self.settings.sort_by == Metric::RSquared {
            self.comparison.reverse();
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for RegressionModel<INPUT, OUTPUT, InputArray, OutputArray>
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
        row_vec.push("METAMODEL".to_string());
        let decider = f64::midpoint(
            self.metamodel.0.mean_train_score(),
            self.metamodel.0.mean_test_score(),
        )
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
