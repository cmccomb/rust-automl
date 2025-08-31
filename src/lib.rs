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
#![warn(missing_docs, rustdoc::missing_doc_code_examples)]
#![doc = include_str!("../README.md")]

pub mod settings;
pub use settings::Settings;
use settings::{Algorithm, FinalAlgorithm, PreProcessing};

pub mod cookbook;

mod utils;
use utils::elementwise_multiply;

use itertools::Itertools;
#[cfg(any(feature = "nd"))]
use ndarray::{Array1, Array2};
use smartcore::api::SupervisedEstimator;
use smartcore::linalg::basic::arrays::{Array1, MutArrayView1, MutArrayView2};
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
    linalg::basic::{
        arrays::{Array, Array2},
        matrix::DenseMatrix,
    },
    model_selection::CrossValidationResult,
};
// use std::any::Any;
// use std::collections::HashSet;
// use std::hash::Hash;
use std::ops::Deref;
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    io::{Read, Write},
    time::Duration,
};
#[cfg(any(feature = "csv"))]
use {
    polars::prelude::{DataFrame, Float32Type},
    utils::validate_and_read,
};

use {
    comfy_table::{
        Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
    },
    humantime::format_duration,
};

#[cfg(any(feature = "csv"))]
impl IntoSupervisedData for (&str, usize) {
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        let (filepath, target_index) = self;
        let df = validate_and_read(filepath);

        // Get target variables
        let target_column_name = df.get_column_names()[target_index];
        let series = df.column(target_column_name).unwrap().clone();
        let target_df = DataFrame::new(vec![series]).unwrap();
        let ndarray = target_df.to_ndarray::<Float32Type>().unwrap();
        let y = ndarray.into_raw_vec();

        // Get the rest of the data
        let features = df.drop(target_column_name).unwrap();
        let (height, width) = features.shape();
        let ndarray = features.to_ndarray::<Float32Type>().unwrap();
        let x = DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap());
        (x, y)
    }
}

#[cfg(any(feature = "csv"))]
impl IntoFeatures for &str {
    fn to_dense_matrix(self) -> DenseMatrix<f32> {
        let df = validate_and_read(self);

        // Get the rest of the data
        let (height, width) = df.shape();
        let ndarray = df.to_ndarray::<Float32Type>().unwrap();
        DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap())
    }
}

#[cfg(any(feature = "nd"))]
impl IntoFeatures for Array2<f32> {
    fn to_dense_matrix(self) -> DenseMatrix<f32> {
        DenseMatrix::from_array(self.shape()[0], self.shape()[1], self.as_slice().unwrap())
    }
}

#[cfg(any(feature = "nd"))]
impl IntoLabels for Array1<f32> {
    fn into_vec(self) -> Vec<f32> {
        self.to_vec()
    }
}

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
    // /// The number of classes in the data.
    // number_of_classes: usize,
    /// The results of the model comparison.
    comparison: Vec<(
        CrossValidationResult,
        Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
        Duration,
    )>,
    /// The final model.
    metamodel: (
        CrossValidationResult,
        FinalAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
        Duration,
    ),
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
    // SupervisedModel<INPUT, OUTPUT, InputArray, OutputArray>:
    //     SupervisedEstimator<InputArray, OutputArray, Box<dyn Any>>,
{
    // /// Create a new supervised model. This function accepts various types of syntax. For instance, it will work for vectors:
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// let model = automl::SupervisedModel::new(
    // ///     (vec![vec![1.0; 5]; 5],
    // ///     vec![1.0; 5]),
    // ///     automl::Settings::default_regression(),
    // /// );
    // /// ```
    // /// It also works for some ndarray datatypes:
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// #[cfg(any(feature = "nd"))]
    // /// let model = SupervisedModel::new(
    // ///     (
    // ///         ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]),
    // ///         ndarray::arr1(&[1.0, 2.0])
    // ///     ),
    // ///     automl::Settings::default_regression(),
    // /// );
    // /// ```
    // /// But you can also create a new supervised model from a [smartcore toy dataset](https://docs.rs/smartcore/0.2.0/smartcore/dataset/index.html)
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// let model = SupervisedModel::new(
    // ///     smartcore::dataset::diabetes::load_dataset(),
    // ///     Settings::default_regression()
    // /// );
    // /// ```
    // /// You can even create a new supervised model directly from a CSV!
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// #[cfg(any(feature = "csv"))]
    // /// let model = SupervisedModel::new(
    // ///     ("data/diabetes.csv", 10),
    // ///     Settings::default_regression()
    // /// );
    // /// ```
    // /// And that CSV can even come from a URL
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// #[cfg(any(feature = "csv"))]
    // /// let mut model = automl::SupervisedModel::new(
    // ///         (
    // ///         "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    // ///         8,
    // ///     ),
    // ///     Settings::default_regression(),
    // /// );
    // pub fn new<D>(data: InputArray, settings: Settings) -> Self
    // where
    //     OUTPUT: From<INPUT>,
    // {
    //     // Get shape of data
    //     let (height, width) = data.shape();
    //
    //     // Get last column as y
    //     let y: Vec<OUTPUT> = (0..height)
    //         .map(|idx| data.get_col(width - 1).get(idx).clone().into())
    //         .collect();
    //
    //     // Get all but last column as x
    //     let x: Vec<Vec<INPUT>> = (0..height)
    //         .map(|idx| {
    //             (0..(width - 1))
    //                 .map(|jdx| data.get_col(jdx).get(idx).clone())
    //                 .collect()
    //         })
    //         .collect();
    //
    //     Self::build(DenseMatrix::from_2d_vec(&x).expect("asdf"), y, settings)
    // }

    // /// Load the supervised model from a file saved previously
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// # let mut model = SupervisedModel::new(
    // /// #    smartcore::dataset::diabetes::load_dataset(),
    // /// #    Settings::default_regression()
    // /// # );
    // /// # model.save("tests/load_that_model.aml");
    // /// let model = SupervisedModel::new_from_file("tests/load_that_model.aml");
    // /// # std::fs::remove_file("tests/load_that_model.aml");
    // /// ```
    // #[must_use]
    // pub fn new_from_file(file_name: &str) -> Self {
    //     let mut buf: Vec<u8> = Vec::new();
    //     std::fs::File::open(file_name)
    //         .and_then(|mut f| f.read_to_end(&mut buf))
    //         .expect("Cannot load model from file.");
    //     bincode::serde::decode_from_slice(&buf, bincode::config::standard())
    //         .expect("Can not deserialize the model")
    //         .0
    // }

    /// Predict values using the final model based on a vec.
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// # let mut model = SupervisedModel::new(
    /// #     smartcore::dataset::diabetes::load_dataset(),
    /// #    Settings::default_regression()
    /// # .only(automl::settings::Algorithm::Linear)
    /// # );
    /// # model.train();
    /// model.predict(vec![vec![5.0; 10]; 5]);
    /// ```
    /// Or predict values using the final model based on ndarray.
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// # #[cfg(any(feature = "nd"))]
    /// # let mut model = SupervisedModel::new(
    /// #     smartcore::dataset::diabetes::load_dataset(),
    /// #     Settings::default_regression()
    /// # .only(automl::settings::Algorithm::Linear)
    /// # );
    /// # #[cfg(any(feature = "nd"))]
    /// # model.train();
    /// #[cfg(any(feature = "nd"))]
    /// model.predict(
    ///     ndarray::arr2(&[
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ///     ])
    /// );
    /// ```
    /// You can also predict from a CSV file
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// # #[cfg(any(feature = "csv"))]
    /// # let mut model = SupervisedModel::new(
    /// #     ("data/diabetes.csv", 10),
    /// #     Settings::default_regression()
    /// # .only(automl::settings::Algorithm::Linear)
    /// # );
    /// # #[cfg(any(feature = "csv"))]
    /// # model.train();
    /// #[cfg(any(feature = "csv"))]
    /// model.predict("data/diabetes_without_target.csv");
    /// ```
    ///
    /// # Panics
    ///
    /// If the model has not been trained, this function will panic.
    pub fn predict(&self, x: InputArray) -> OutputArray {
        let x = &self.preprocess(x);
        let top_model = self.comparison[0].1.clone();
        match &self.settings.final_model_approach {
            FinalAlgorithm::None => panic!(""),
            FinalAlgorithm::Best => self.predict_by_model(x, top_model),
            FinalAlgorithm::Blending { .. } => self.predict_by_model(x, top_model), //self.predict_blended_model(x, algorithm),
        }
    }

    /// Runs a model comparison and trains a final model.
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// # .only(automl::settings::Algorithm::Linear)
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

        // // Preprocess the data
        // self.x_train = self.preprocess(self.x_train.clone());

        // Split validatino out if blending
        // if let FinalAlgorithm::Blending {
        //     meta_training_fraction: fraction,
        //     meta_testing_fraction: _,
        //     algorithm: _,
        // } = &self.settings.final_model_approach
        // {
        //     let (x_train, x_val, y_train, y_val) = train_test_split(
        //         &self.x_train,
        //         &self.y_train,
        //         fraction.clone(),
        //         self.settings.shuffle,
        //         None,
        //     );
        //     self.x_train = x_train;
        //     self.y_train = y_train;
        //     self.y_val = y_val;
        //     self.x_val = x_val;
        // }

        // // Run logistic regression
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::LogisticRegression)
        // {
        //     self.record_model(LogisticRegressionWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // // Run random forest classification
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::RandomForestClassifier)
        // {
        //     self.record_model(RandomForestClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // // Run k-nearest neighbor classifier
        // if !self.settings.skiplist.contains(&Algorithm::KNNClassifier) {
        //     self.record_model(KNNClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // // Run decision tree classification
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::DecisionTreeClassifier)
        // {
        //     self.record_model(DecisionTreeClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::GaussianNaiveBayes)
        // {
        //     self.record_model(GaussianNaiveBayesClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::CategoricalNaiveBayes)
        //     && std::mem::discriminant(&self.settings.preprocessing)
        //         != std::mem::discriminant(&PreProcessing::ReplaceWithPCA {
        //             number_of_components: 1,
        //         })
        //     && std::mem::discriminant(&self.settings.preprocessing)
        //         != std::mem::discriminant(&PreProcessing::ReplaceWithSVD {
        //             number_of_components: 1,
        //         })
        // {
        //     self.record_model(CategoricalNaiveBayesClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        //
        // if self.number_of_classes == 2 && !self.settings.skiplist.contains(&Algorithm::SVC) {
        //     self.record_model(SupportVectorClassifierWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }

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

        //
        // if !self.settings.skiplist.contains(&Algorithm::Linear) {
        //     self.record_trained_model(Algorithms::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self.settings.skiplist.contains(&Algorithm::SVR) {
        //     self.record_trained_model(SupportVectorRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self.settings.skiplist.contains(&Algorithm::Lasso) {
        //     self.record_trained_model(RidgeRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self.settings.skiplist.contains(&Algorithm::Ridge) {
        //     self.record_trained_model(LassoRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self.settings.skiplist.contains(&Algorithm::ElasticNet) {
        //     self.record_trained_model(ElasticNetRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::DecisionTreeRegressor)
        // {
        //     self.record_trained_model(DecisionTreeRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self
        //     .settings
        //     .skiplist
        //     .contains(&Algorithm::RandomForestRegressor)
        // {
        //     self.record_trained_model(RandomForestRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }
        //
        // if !self.settings.skiplist.contains(&Algorithm::KNNRegressor) {
        //     self.record_trained_model(KNNRegressorWrapper::cv_model(
        //         &self.x_train,
        //         &self.y_train,
        //         &self.settings,
        //     ));
        // }

        // if let FinalAlgorithm::Blending {
        //     algorithm,
        //     meta_training_fraction,
        //     meta_testing_fraction,
        // } = self.settings.final_model_approach
        // {
        //     self.train_blended_model(algorithm, meta_training_fraction, meta_testing_fraction);
        // }
    }

    // /// Save the supervised model to a file for later use
    // /// ```
    // /// # use automl::{SupervisedModel, Settings};
    // /// let mut model = SupervisedModel::new(
    // ///     smartcore::dataset::diabetes::load_dataset(),
    // ///     Settings::default_regression()
    // /// );
    // /// model.save("tests/save_that_model.aml");
    // /// # std::fs::remove_file("tests/save_that_model.aml");
    // /// ```
    // pub fn save(&self, file_name: &str) {
    //     let serial = bincode::serde::encode_to_vec(&self, bincode::config::standard())
    //         .expect("Cannot serialize model.");
    //     std::fs::File::create(file_name)
    //         .and_then(|mut f| f.write_all(&serial))
    //         .expect("Cannot write model to file.");
    // }

    // /// Save the best model for later use as a smartcore native object.
    // /// ```
    // /// # use automl::{SupervisedModel, Settings, settings::Algorithm};
    // /// use std::io::Read;
    // ///
    // /// let mut model = SupervisedModel::new(
    // ///     smartcore::dataset::diabetes::load_dataset(),
    // ///     Settings::default_regression()
    // /// # .only(Algorithm::Linear)
    // /// );
    // /// model.train();
    // /// model.save("tests/save_best.sc");
    // /// # std::fs::remove_file("tests/save_best.sc");
    // /// ```
    // pub fn save_best(&self, file_name: &str) {
    //     if matches!(self.settings.final_model_approach, FinalAlgorithm::Best) {
    //         std::fs::File::create(file_name)
    //             .and_then(|mut f| f.write_all(&self.comparison[0].model))
    //             .expect("Cannot write model to file.");
    //     }
    // }
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
    pub fn build(
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
                FinalAlgorithm::default_blending(),
                Duration::default(),
            ),
            preprocessing_pca: None,
            preprocessing_svd: None,
        }
    }

    // /// Train the supervised model.
    // ///
    // /// # Arguments
    // ///
    // /// * `algo` - The algorithm to use
    // /// * `training_fraction` - The fraction of the data to use for training
    // /// * `testing_fraction` - The fraction of the data to use for testing
    // fn train_blended_model(
    //     &mut self,
    //     algo: Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    //     training_fraction: f32,
    //     testing_fraction: f32,
    // ) {
    //     // Make the data
    //     let mut meta_x: Vec<Vec<f32>> = Vec::new();
    //     for model in &self.comparison {
    //         meta_x.push(self.predict_by_model(&self.x_val, model));
    //     }
    //     let xdm = DenseMatrix::from_2d_vec(&meta_x)
    //         .expect("Could not convert 2dvec to densematrix")
    //         .transpose();
    //
    //     // Split into datasets
    //     let (x_train, x_test, y_train, y_test) = train_test_split(
    //         &xdm,
    //         &self.y_val,
    //         training_fraction / (training_fraction + testing_fraction),
    //         self.settings.shuffle,
    //         None,
    //     );
    //
    //     // Train the model
    //     // let model = LassoRegressorWrapper::train(&x_train, &y_train, &self.settings);
    //     let model = algo.get_trainer()(&x_train, &y_train, &self.settings);
    //
    //     // Score the model
    //     let train_score = self.settings.get_metric()(
    //         &y_train,
    //         &algo.get_predictor()(&x_train, &model, &self.settings),
    //         // &LassoRegressorWrapper::predict(&x_train, &model, &self.settings),
    //     );
    //     let test_score = self.settings.get_metric()(
    //         &y_test,
    //         &algo.get_predictor()(&x_test, &model, &self.settings),
    //         // &LassoRegressorWrapper::predict(&x_test, &model, &self.settings),
    //     );
    //
    //     self.metamodel = (
    //         CrossValidationResult {
    //             test_score: vec![test_score as f64; 1],
    //             train_score: vec![train_score as f64; 1],
    //         },
    //         model,
    //         Duration::default(),
    //     );
    // }
    //
    // /// Predict using all of the trained models.
    // ///
    // /// # Arguments
    // ///
    // /// * `x` - The input data
    // /// * `algo` - The algorithm to use
    // ///
    // /// # Returns
    // ///
    // /// * The predicted values
    // fn predict_blended_model(
    //     &self,
    //     x: &InputArray,
    //     algo: Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    // ) -> OutputArray {
    //     // Make the data
    //     let mut meta_x: Vec<Vec<f32>> = Vec::new();
    //     for i in 0..self.comparison.len() {
    //         let model = &self.comparison[i];
    //         meta_x.push(self.predict_by_model(x, model));
    //     }
    //
    //     //
    //     let xdm = DenseMatrix::from_2d_vec(&meta_x)
    //         .expect("Could not convert 2dvec to DenseMatrix")
    //         .transpose();
    //     let metamodel = &self.metamodel.1;
    //
    //     // Train the model
    //     algo.get_predictor()(&xdm, metamodel, &self.settings)
    // }

    /// Predict using a single model.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `model` - The model to use
    ///
    /// # Returns
    ///
    /// * The predicted values
    fn predict_by_model(
        &self,
        x: &InputArray,
        model: Algorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> OutputArray {
        model.predict(x)
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
        // if self.settings.sort_by == Metric::RSquared || self.settings.sort_by == Metric::Accuracy {
        //     self.comparison.reverse();
        // }
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
//
// /// This contains the results of a single model
// #[derive(serde::Serialize, serde::Deserialize)]
// struct Model {
//     /// The cross validation score of the model
//     #[serde(with = "CrossValidationResultDef")]
//     score: CrossValidationResult,
//     /// The algorithm used
//     name: Algorithm,
//     /// The time it took to train the model
//     duration: Duration,
//     /// The model
//     model: Vec<u8>,
// }

// impl Default for Model {
//     fn default() -> Self {
//         Self {
//             score: CrossValidationResult {
//                 test_score: vec![],
//                 train_score: vec![],
//             },
//             name: Algorithm::Linear,
//             duration: Duration::default(),
//             model: vec![],
//         }
//     }
// }

/// This is a wrapper for the `CrossValidationResult`
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "CrossValidationResult")]
struct CrossValidationResultDef {
    /// Vector with test scores on each cv split
    pub test_score: Vec<f64>,
    /// Vector with training scores on each cv split
    pub train_score: Vec<f64>,
}
