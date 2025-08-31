#![deny(clippy::correctness)]
#![warn(
    clippy::all,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items
)]
#![allow(clippy::module_name_repetitions, clippy::too_many_lines)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod settings;
pub use settings::Settings;
use settings::{Algorithm, FinalAlgorithm, PreProcessing};

pub mod cookbook;

mod utils;
use utils::elementwise_multiply;

use itertools::Itertools;
use smartcore::linalg::basic::arrays::{Array1, MutArrayView1};
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::evd::EVDDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;
use smartcore::{
    decomposition::{
        pca::{PCA, PCAParameters},
        svd::{SVD, SVDParameters},
    },
    linalg::basic::arrays::{Array, Array2},
    model_selection::CrossValidationResult,
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    time::Duration,
};

pub use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::settings::Metric;
use {
    comfy_table::{
        Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
    },
    humantime::format_duration,
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
    /// # use automl::{settings, SupervisedModel, Settings};
    /// # let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64; 6]; 16]).unwrap();
    /// # let y = vec![0.0_f64; 16];
    /// # let mut model = SupervisedModel::new(
    /// #    x,
    /// #    y,
    /// #    Settings::default_regression()
    /// #        .only(settings::Algorithm::default_linear()),
    /// # );
    /// # model.train();
    /// let X = DenseMatrix::from_2d_vec(&vec![vec![5.0; 6]; 5]).unwrap();
    /// model.predict(X);
    /// ```
    /// # Panics
    ///
    /// If the model has not been trained, this function will panic.
    pub fn predict(self, x: InputArray) -> OutputArray {
        let x = self.preprocess(x);
        match self.settings.final_model_approach {
            FinalAlgorithm::None => panic!(""),
            FinalAlgorithm::Best => match &self.comparison.get(0).expect("").1 {
                Algorithm::Linear(model) => model.predict(&x),
                Algorithm::Lasso(model) => model.predict(&x),
                Algorithm::Ridge(model) => model.predict(&x),
                Algorithm::ElasticNet(model) => model.predict(&x),
                Algorithm::RandomForestRegressor(model) => model.predict(&x),
                Algorithm::DecisionTreeRegressor(model) => model.predict(&x),
            }.expect(
                "Error during inference. This is likely a bug in the AutoML library. Please open an issue on GitHub.",
            ) // FinalAlgorithm::Blending { .. } => self.predict_by_model(x, top_model), //self.predict_blended_model(x, algorithm),
        }
    }

    /// Runs a model comparison and trains a final model.
    /// ```
    /// # use automl::{settings, SupervisedModel, Settings};
    /// # use smartcore::linalg::basic::matrix::DenseMatrix;
    /// # let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64; 6]; 16]).unwrap();
    /// # let y = vec![0.0_f64; 16];
    /// let mut model = SupervisedModel::new(
    ///     x,
    ///     y,
    ///     Settings::default_regression()
    /// #        .only(settings::Algorithm::default_linear())
    /// );
    /// model.train();
    /// ```
    pub fn train(&mut self) {
        // Train any necessary preprocessing
        if let PreProcessing::ReplaceWithPCA {
            number_of_components,
        } = self.settings.preprocessing
        {
            self.train_pca(&self.x_train.clone(), number_of_components);
        }
        if let PreProcessing::ReplaceWithSVD {
            number_of_components,
        } = self.settings.preprocessing
        {
            self.train_svd(&self.x_train.clone(), number_of_components);
        }

        // Iterate over variants in Algorithm
        for alg in Algorithm::all_algorithms() {
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

/// Private functions go here
impl<INPUT, OUTPUT, InputArray, OutputArray> SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>
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
        settings: Settings<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        Self {
            settings,
            x_train: x.clone(),
            // number_of_classes: Self::count_classes(&y),
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

    /// Get interaction features for the data.
    ///
    /// # Arguments
    fn interaction_features(mut x: InputArray) -> InputArray {
        let (height, width) = x.shape();
        for column_1 in 0..width {
            for column_2 in (column_1 + 1)..width {
                let col1: Vec<INPUT> = (0..height)
                    .map(|idx| x.get_col(column_1).get(idx).clone())
                    .collect();
                let col2: Vec<INPUT> = (0..height)
                    .map(|idx| x.get_col(column_2).get(idx).clone())
                    .collect();
                let feature = elementwise_multiply(&col1, &col2);
                let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                    .expect("Cannot create matrix")
                    .transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    /// Get polynomial features for the data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `order` - The order of the polynomial
    ///
    /// # Returns
    ///
    /// * The data with polynomial features
    fn polynomial_features(mut x: InputArray, order: usize) -> InputArray {
        // Get the shape of the matrix
        let (height, width) = x.shape();

        // For each order, get the combinations of columns with replacement
        for n in 2..=order {
            // Get combinations of columns with replacement
            let combinations = (0..width).combinations_with_replacement(n);

            // For each combination, multiply the columns together and add to the matrix
            for combo in combinations {
                // Start with a vector of ones
                let mut feature: Vec<INPUT> = vec![INPUT::one(); height];

                // Multiply the columns together
                for column in combo {
                    let col: Vec<INPUT> = (0..height)
                        .map(|idx| x.get_col(column).get(idx).clone())
                        .collect();
                    feature = elementwise_multiply(&col, &feature);
                }

                // Add the new column to the matrix
                let new_column = DenseMatrix::from_2d_vec(&vec![feature; 1])
                    .expect("Cannot create matrix")
                    .transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    /// Train PCA on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    fn train_pca(&mut self, x: &InputArray, n: usize) {
        let pca = PCA::fit(
            x,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .unwrap();
        self.preprocessing_pca = Some(pca);
    }

    /// Get PCA features for the data using the trained PCA preprocessor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    fn pca_features(&self, x: &InputArray, _: usize) -> InputArray {
        self.preprocessing_pca
            .as_ref()
            .unwrap()
            .transform(x)
            .expect("Could not transform data using PCA")
    }

    /// Train SVD on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    fn train_svd(&mut self, x: &InputArray, n: usize) {
        let svd = SVD::fit(x, SVDParameters::default().with_n_components(n)).unwrap();
        self.preprocessing_svd = Some(svd);
    }

    /// Get SVD features for the data.
    fn svd_features(&self, x: &InputArray, _: usize) -> InputArray {
        self.preprocessing_svd
            .as_ref()
            .unwrap()
            .transform(x)
            .expect("Could not transform data using SVD")
    }

    /// Pre process the data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    ///
    /// # Returns
    ///
    /// * The preprocessed data
    fn preprocess(&self, x: InputArray) -> InputArray {
        match self.settings.preprocessing {
            PreProcessing::None => x,
            PreProcessing::AddInteractions => Self::interaction_features(x),
            PreProcessing::AddPolynomial { order } => Self::polynomial_features(x, order),
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => self.pca_features(&x, number_of_components),
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => self.svd_features(&x, number_of_components),
        }
    }

    /// Count the number of classes in the data.
    ///
    /// # Arguments
    ///
    /// * `y` - The data to count the classes in
    ///
    /// # Returns
    ///
    /// * The number of classes
    // fn count_classes(y: &OutputArray) -> usize {
    //     let mut classes = HashSet::new();
    //     for value in y.iterator(0_u8) {
    //         classes.insert(value.clone());
    //     }
    //     classes.len()
    // }

    /// Record a model in the comparison.
    fn record_trained_model(
        &mut self,
        trained_model: (
            CrossValidationResult,
            Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
            Duration,
        ),
    ) {
        self.comparison.push(trained_model);
        self.sort();
    }

    /// Sort the models in the comparison by their mean test scores.
    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            a.0.mean_test_score()
                .partial_cmp(&b.0.mean_test_score())
                .unwrap_or(Equal)
        });
        if self.settings.sort_by == Metric::RSquared {
            self.comparison.reverse();
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
