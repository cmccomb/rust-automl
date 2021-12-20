#![warn(clippy::all)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(clippy::missing_docs_in_private_items)]
#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod settings;
pub use settings::Settings;
use settings::{Algorithm, Distance, Kernel, Metric, PreProcessing};

mod algorithms;
use algorithms::{
    CategoricalNaiveBayesClassifierWrapper, DecisionTreeClassifierWrapper,
    DecisionTreeRegressorWrapper, ElasticNetRegressorWrapper, GaussianNaiveBayesClassifierWrapper,
    KNNClassifierWrapper, KNNRegressorWrapper, LassoRegressorWrapper, LinearRegressorWrapper,
    LogisticRegressionWrapper, ModelWrapper, RandomForestClassifierWrapper,
    RandomForestRegressorWrapper, RidgeRegressorWrapper, SupportVectorClassifierWrapper,
    SupportVectorRegressorWrapper,
};

mod utils;
use utils::elementwise_multiply;

use itertools::Itertools;
use smartcore::{
    dataset::Dataset,
    decomposition::{
        pca::{PCAParameters, PCA},
        svd::{SVDParameters, SVD},
    },
    linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix},
    model_selection::{train_test_split, CrossValidationResult},
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
    time::Duration,
};

#[cfg(any(feature = "nd"))]
use ndarray::{Array1, Array2};

#[cfg(any(feature = "gui"))]
use eframe::{egui, epi};

#[cfg(any(feature = "csv"))]
use polars::prelude::{CsvReader, DataFrame, Float32Type, SerReader};

#[cfg(any(feature = "display"))]
use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};

use crate::settings::FinalModel;
#[cfg(any(feature = "display"))]
use humantime::format_duration;

/// Trains and compares supervised models
pub struct SupervisedModel {
    settings: Settings,
    x_train: DenseMatrix<f32>,
    y_train: Vec<f32>,
    x_val: DenseMatrix<f32>,
    y_val: Vec<f32>,
    number_of_classes: usize,
    comparison: Vec<Model>,
    metamodel: Model,
    preprocessing: (
        Option<PCA<f32, DenseMatrix<f32>>>,
        Option<SVD<f32, DenseMatrix<f32>>>,
    ),
    #[cfg(any(feature = "gui"))]
    current_x: Vec<f32>,
}

impl SupervisedModel {
    /// Create a new supervised model from a [smartcore toy dataset](https://docs.rs/smartcore/0.2.0/smartcore/dataset/index.html)
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// ```
    pub fn new_from_dataset(dataset: Dataset<f32, f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        let y = dataset.target;

        Self {
            settings,
            x_train: x.clone(),
            y_train: y.clone(),
            x_val: DenseMatrix::new(0, 0, vec![]),
            y_val: vec![],
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            #[cfg(any(feature = "gui"))]
            current_x: vec![0.0; x.shape().1],
            preprocessing: (None, None),
            metamodel: Default::default(),
        }
    }

    /// Create a new supervised model using vec data
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = automl::SupervisedModel::new_from_vec(
    ///     vec![vec![1.0; 5]; 5],
    ///     vec![1.0; 5],
    ///     automl::Settings::default_regression(),
    /// );    
    /// ```
    pub fn new_from_vec(x: Vec<Vec<f32>>, y: Vec<f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_2d_vec(&x);

        Self {
            settings,
            x_train: x.clone(),
            y_train: y.clone(),
            x_val: DenseMatrix::new(0, 0, vec![]),
            y_val: vec![],
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            #[cfg(any(feature = "gui"))]
            current_x: vec![0.0; x.shape().1],
            preprocessing: (None, None),
            metamodel: Default::default(),
        }
    }

    /// Create a new supervised model using ndarray data
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// use ndarray::{arr1, arr2};
    /// let model = automl::SupervisedModel::new_from_ndarray(
    ///     arr2(&[[1.0, 2.0], [3.0, 4.0]]),
    ///     arr1(&[1.0, 2.0]),
    ///     automl::Settings::default_regression(),
    /// );
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "nd")))]
    #[cfg(any(feature = "nd"))]
    pub fn new_from_ndarray(x: Array2<f32>, y: Array1<f32>, settings: Settings) -> Self {
        let x = DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap());
        let y = y.to_vec();

        Self {
            settings,
            x_train: x.clone(),
            y_train: y.clone(),
            x_val: DenseMatrix::new(0, 0, vec![]),
            y_val: vec![],
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            #[cfg(any(feature = "gui"))]
            current_x: vec![0.0; x.shape().1],
            preprocessing: (None, None),
            metamodel: Default::default(),
        }
    }

    /// Create a new supervised model from a csv
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = SupervisedModel::new_from_csv(
    ///     "data/diabetes.csv",
    ///     10,
    ///     true,
    ///     Settings::default_regression()
    /// );
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "csv")))]
    #[cfg(any(feature = "csv"))]
    pub fn new_from_csv(
        filepath: &str,
        target_index: usize,
        header: bool,
        settings: Settings,
    ) -> Self {
        let df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

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

        Self {
            settings,
            x_train: x.clone(),
            y_train: y.clone(),
            x_val: DenseMatrix::new(0, 0, vec![]),
            y_val: vec![],
            number_of_classes: Self::count_classes(&y),
            comparison: vec![],
            #[cfg(any(feature = "gui"))]
            current_x: vec![0.0; x.shape().1],
            preprocessing: (None, None),
            metamodel: Default::default(),
        }
    }

    /// Runs a model comparison and trains a final model.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.train();
    /// ```
    pub fn train(&mut self) {
        // Preprocess the data
        self.x_train = self.preprocess(self.x_train.clone());

        // Split validatino out if blending
        match &self.settings.final_model_approach {
            FinalModel::None => {}
            FinalModel::Best => {}
            FinalModel::Blending {
                meta_training_fraction,
                meta_testing_fraction: _,
                algorithm: _,
            } => {
                let (x_train, x_val, y_train, y_val) = train_test_split(
                    &self.x_train,
                    &self.y_train,
                    *meta_training_fraction,
                    self.settings.shuffle,
                );
                self.x_train = x_train;
                self.y_train = y_train;
                self.y_val = y_val;
                self.x_val = x_val;
            }
        }

        // Run logistic regression
        if !self
            .settings
            .skiplist
            .contains(&Algorithm::LogisticRegression)
        {
            self.record_model(LogisticRegressionWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        // Run random forest classification
        if !self
            .settings
            .skiplist
            .contains(&Algorithm::RandomForestClassifier)
        {
            self.record_model(RandomForestClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        // Run k-nearest neighbor classifier
        if !self.settings.skiplist.contains(&Algorithm::KNNClassifier) {
            self.record_model(KNNClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::DecisionTreeClassifier)
        {
            self.record_model(DecisionTreeClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::GaussianNaiveBayes)
        {
            self.record_model(GaussianNaiveBayesClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::CategoricalNaiveBayes)
        {
            self.record_model(CategoricalNaiveBayesClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if self.number_of_classes == 2 && !self.settings.skiplist.contains(&Algorithm::SVC) {
            self.record_model(SupportVectorClassifierWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::Linear) {
            self.record_model(LinearRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::SVR) {
            self.record_model(SupportVectorRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::Lasso) {
            self.record_model(RidgeRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::Ridge) {
            self.record_model(LassoRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::ElasticNet) {
            self.record_model(ElasticNetRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::DecisionTreeRegressor)
        {
            self.record_model(DecisionTreeRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self
            .settings
            .skiplist
            .contains(&Algorithm::RandomForestRegressor)
        {
            self.record_model(RandomForestRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        if !self.settings.skiplist.contains(&Algorithm::KNNRegressor) {
            self.record_model(KNNRegressorWrapper::cv_model(
                &self.x_train,
                &self.y_train,
                &self.settings,
            ));
        }

        match self.settings.final_model_approach {
            FinalModel::Blending {
                algorithm,
                meta_training_fraction,
                meta_testing_fraction,
            } => self.train_blended_model(algorithm, meta_training_fraction, meta_testing_fraction),
            _ => {}
        }
    }

    /// Predict values using the final model based on a vec.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.train();
    /// model.predict_from_vec(vec![vec![5.0; 10]; 5]);
    /// ```
    pub fn predict_from_vec(&mut self, x: Vec<Vec<f32>>) -> Vec<f32> {
        self.predict(&DenseMatrix::from_2d_vec(&x))
    }

    /// Predict values using the final model based on ndarray.
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// use ndarray::arr2;
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.train();
    /// model.predict_from_ndarray(
    ///     arr2(&[
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ///         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ///     ])
    /// );
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "nd")))]
    #[cfg(any(feature = "nd"))]
    pub fn predict_from_ndarray(&mut self, x: Array2<f32>) -> Vec<f32> {
        self.predict(&DenseMatrix::from_array(
            x.shape()[0],
            x.shape()[1],
            x.as_slice().unwrap(),
        ))
    }

    /// Runs an interactive GUI to demonstrate the final model
    /// ```no_run
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new_from_dataset(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.train();
    /// model.run_gui();
    /// ```
    /// ![Example of interactive gui demo](https://raw.githubusercontent.com/cmccomb/rust-automl/master/assets/gui.png)
    #[cfg_attr(docsrs, doc(cfg(feature = "gui")))]
    #[cfg(any(feature = "gui"))]
    pub fn run_gui(self) {
        let native_options = eframe::NativeOptions::default();
        eframe::run_native(Box::new(self), native_options);
    }
}

/// Private functions go here
impl SupervisedModel {
    fn train_blended_model(
        &mut self,
        algo: Algorithm,
        training_fraction: f32,
        testing_fraction: f32,
    ) {
        // Make the data
        let mut meta_x: Vec<Vec<f32>> = Vec::new();
        for model in &self.comparison {
            meta_x.push(self.predict_by_model(&self.x_val, model))
        }
        let xdm = DenseMatrix::from_2d_vec(&meta_x).transpose();

        // Split into datasets
        let (x_train, x_test, y_train, y_test) = train_test_split(
            &xdm,
            &self.y_val,
            training_fraction / (training_fraction + testing_fraction),
            self.settings.shuffle,
        );

        // Train the model
        // let model = LassoRegressorWrapper::train(&x_train, &y_train, &self.settings);
        let model = (*algo.get_trainer())(&x_train, &y_train, &self.settings);

        // Score the model
        let train_score = (*self.settings.get_metric())(
            &y_train,
            &(*algo.get_predictor())(&x_train, &model, &self.settings),
            // &LassoRegressorWrapper::predict(&x_train, &model, &self.settings),
        );
        let test_score = (*self.settings.get_metric())(
            &y_test,
            &(*algo.get_predictor())(&x_test, &model, &self.settings),
            // &LassoRegressorWrapper::predict(&x_test, &model, &self.settings),
        );

        self.metamodel = Model {
            score: CrossValidationResult {
                test_score: vec![test_score; 1],
                train_score: vec![train_score; 1],
            },
            name: algo,
            duration: Default::default(),
            model,
        };
    }

    fn predict_blended_model(&mut self, x: &DenseMatrix<f32>, algo: Algorithm) -> Vec<f32> {
        // Make the data
        let mut meta_x: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.comparison.len() {
            let model = &self.comparison[i];
            meta_x.push(self.predict_by_model(&x, model))
        }

        //
        let xdm = DenseMatrix::from_2d_vec(&meta_x).transpose();
        let metamodel = &self.metamodel.model;

        // Train the model
        (*algo.get_predictor())(&xdm, metamodel, &self.settings)
    }

    fn predict_by_model(&self, x: &DenseMatrix<f32>, model: &Model) -> Vec<f32> {
        let saved_model = &model.model;
        match model.name {
            Algorithm::Linear => LinearRegressorWrapper::predict(x, saved_model, &self.settings),
            Algorithm::Lasso => LassoRegressorWrapper::predict(x, saved_model, &self.settings),
            Algorithm::Ridge => RidgeRegressorWrapper::predict(x, saved_model, &self.settings),
            Algorithm::ElasticNet => {
                ElasticNetRegressorWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::RandomForestRegressor => {
                RandomForestRegressorWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::KNNRegressor => KNNRegressorWrapper::predict(x, saved_model, &self.settings),
            Algorithm::SVR => {
                SupportVectorRegressorWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::DecisionTreeRegressor => {
                DecisionTreeRegressorWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::LogisticRegression => {
                LogisticRegressionWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::RandomForestClassifier => {
                RandomForestClassifierWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::DecisionTreeClassifier => {
                DecisionTreeClassifierWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::KNNClassifier => {
                KNNClassifierWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::SVC => {
                SupportVectorClassifierWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::GaussianNaiveBayes => {
                GaussianNaiveBayesClassifierWrapper::predict(x, saved_model, &self.settings)
            }
            Algorithm::CategoricalNaiveBayes => {
                CategoricalNaiveBayesClassifierWrapper::predict(x, saved_model, &self.settings)
            }
        }
    }

    fn predict(&mut self, x: &DenseMatrix<f32>) -> Vec<f32> {
        let x = &self.preprocess(x.clone());
        match self.settings.final_model_approach {
            FinalModel::None => panic!(""),
            FinalModel::Best => self.predict_by_model(x, &self.comparison[0]),
            FinalModel::Blending { algorithm, .. } => self.predict_blended_model(x, algorithm),
        }
    }

    fn interaction_features(mut x: DenseMatrix<f32>) -> DenseMatrix<f32> {
        let (_, width) = x.shape();
        for i in 0..width {
            for j in (i + 1)..width {
                let feature = elementwise_multiply(&x.get_col_as_vec(i), &x.get_col_as_vec(j));
                let new_column = DenseMatrix::from_row_vector(feature).transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    fn polynomial_features(mut x: DenseMatrix<f32>, order: usize) -> DenseMatrix<f32> {
        let (height, width) = x.shape();
        for n in 2..=order {
            let combinations = (0..width).into_iter().combinations_with_replacement(n);
            for combo in combinations {
                let mut feature = vec![1.0; height];
                for column in combo {
                    feature = elementwise_multiply(&x.get_col_as_vec(column), &feature);
                }
                let new_column = DenseMatrix::from_row_vector(feature).transpose();
                x = x.h_stack(&new_column);
            }
        }
        x
    }

    fn pca_features(&mut self, x: DenseMatrix<f32>, n: usize) -> DenseMatrix<f32> {
        if let None = self.preprocessing.0 {
            let pca = PCA::fit(
                &x,
                PCAParameters::default()
                    .with_n_components(n)
                    .with_use_correlation_matrix(true),
            )
            .unwrap();
            self.preprocessing.0 = Some(pca);
        }
        self.preprocessing
            .0
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    fn svd_features(&mut self, x: DenseMatrix<f32>, n: usize) -> DenseMatrix<f32> {
        if let None = self.preprocessing.1 {
            let svd = SVD::fit(&x, SVDParameters::default().with_n_components(n)).unwrap();
            self.preprocessing.1 = Some(svd);
        }
        self.preprocessing
            .1
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    fn preprocess(&mut self, x: DenseMatrix<f32>) -> DenseMatrix<f32> {
        match self.settings.preprocessing {
            PreProcessing::None => x,
            PreProcessing::AddInteractions => SupervisedModel::interaction_features(x),
            PreProcessing::AddPolynomial { order } => {
                SupervisedModel::polynomial_features(x, order)
            }
            PreProcessing::ReplaceWithPCA {
                number_of_components,
            } => self.pca_features(x, number_of_components),
            PreProcessing::ReplaceWithSVD {
                number_of_components,
            } => self.svd_features(x, number_of_components),
        }
    }

    fn count_classes(y: &Vec<f32>) -> usize {
        let mut sorted_targets = y.clone();
        sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
        sorted_targets.dedup();
        sorted_targets.len()
    }

    fn record_model(&mut self, model: (CrossValidationResult<f32>, Algorithm, Duration, Vec<u8>)) {
        self.comparison.push(Model {
            score: model.0,
            name: model.1,
            duration: model.2,
            model: model.3,
        });
        self.sort();
    }

    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            a.score
                .mean_test_score()
                .partial_cmp(&b.score.mean_test_score())
                .unwrap_or(Equal)
        });
        if self.settings.sort_by == Metric::RSquared || self.settings.sort_by == Metric::Accuracy {
            self.comparison.reverse();
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "display")))]
#[cfg(any(feature = "display"))]
impl Display for SupervisedModel {
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
            row_vec.push(format!("{}", &model.name));
            row_vec.push(format!("{}", format_duration(model.duration)));
            let decider =
                ((model.score.mean_train_score() + model.score.mean_test_score()) / 2.0).abs();
            if decider > 0.01 && decider < 1000.0 {
                row_vec.push(format!("{:.2}", &model.score.mean_train_score()));
                row_vec.push(format!("{:.2}", &model.score.mean_test_score()));
            } else {
                row_vec.push(format!("{:.3e}", &model.score.mean_train_score()));
                row_vec.push(format!("{:.3e}", &model.score.mean_test_score()));
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
        row_vec.push(format!("{}", self.metamodel.name));
        let decider = ((self.metamodel.score.mean_train_score()
            + self.metamodel.score.mean_test_score())
            / 2.0)
            .abs();
        if decider > 0.01 && decider < 1000.0 {
            row_vec.push(format!("{:.2}", self.metamodel.score.mean_train_score()));
            row_vec.push(format!("{:.2}", self.metamodel.score.mean_test_score()));
        } else {
            row_vec.push(format!("{:.3e}", self.metamodel.score.mean_train_score()));
            row_vec.push(format!("{:.3e}", self.metamodel.score.mean_test_score()));
        }

        // Add row to table
        meta_table.add_row(row_vec);

        // Write
        write!(f, "{}\n{}", table, meta_table)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "gui")))]
#[cfg(any(feature = "gui"))]
impl epi::App for SupervisedModel {
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Add a heading that displays the type of model this is
            ui.heading(format!("{}", self.comparison[0].name));

            // Add a label that shows the prediction
            ui.label(format!(
                "Prediction: y = {}",
                self.predict(&DenseMatrix::from_2d_vec(&vec![self.current_x.to_vec(); 1]))[0]
            ));

            // Separating the model name and prediction from the input values
            ui.separator();

            // Step through input values to make sliders
            for i in 0..self.current_x.len() {
                // Figure out the maximum in the training dataa
                let maxx = self
                    .x_train
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::max);

                // Figure out the minimum in the training data
                let minn = self
                    .x_train
                    .get_col_as_vec(i)
                    .iter()
                    .cloned()
                    .fold(0. / 0., f32::min);

                // Add the slider
                ui.add(
                    egui::Slider::new(&mut self.current_x[i], minn..=maxx).text(format!("x_{}", i)),
                );
            }
        });
    }

    fn name(&self) -> &str {
        "Model Demo"
    }
}

/// This contains the results of a single model
struct Model {
    score: CrossValidationResult<f32>,
    name: Algorithm,
    duration: Duration,
    model: Vec<u8>,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            score: CrossValidationResult {
                test_score: vec![],
                train_score: vec![],
            },
            name: Algorithm::Linear,
            duration: Duration::default(),
            model: vec![],
        }
    }
}
