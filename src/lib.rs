#![warn(
    clippy::all,
    clippy::nursery,
)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(clippy::missing_docs_in_private_items)]
#![doc = include_str!("../README.md")]

pub mod settings;
pub use settings::Settings;
use settings::{Algorithm, Distance, FinalModel, Kernel, Metric, PreProcessing};

pub mod cookbook;

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
    io::{Read, Write},
    time::Duration,
};

#[cfg(any(feature = "nd"))]
use ndarray::{Array1, Array2};

#[cfg(any(feature = "csv"))]
use {
    polars::prelude::{DataFrame, Float32Type},
    utils::validate_and_read,
};

use {
    comfy_table::{
        modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
    },
    humantime::format_duration,
};

/// This trait must be implemented for any types passed to the `SupervisedModel::new` as data.
pub trait IntoSupervisedData {
    /// Converts the struct into paired features and labels
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>);
}

/// Types that implement this trait can be paired in a tuple with a type implementing `IntoLabels` to
/// automatically satisfy `IntoSupervisedData`. This trait is also required for data that is passed to `predict`.
pub trait IntoFeatures {
    /// Converts the struct into a dense matrix of features
    fn to_dense_matrix(self) -> DenseMatrix<f32>;
}

/// Types that implement this trait can be paired in a tuple with a type implementing `IntoFeatures`
/// to automatically satisfy `IntoSupervisedData`.
pub trait IntoLabels {
    /// Converts the struct into a vector of labels
    fn into_vec(self) -> Vec<f32>;
}

impl IntoSupervisedData for Dataset<f32, f32> {
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        (
            DenseMatrix::from_array(self.num_samples, self.num_features, &self.data),
            self.target,
        )
    }
}

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

impl<X, Y> IntoSupervisedData for (X, Y)
where
    X: IntoFeatures,
    Y: IntoLabels,
{
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        (self.0.to_dense_matrix(), self.1.into_vec())
    }
}

impl IntoFeatures for Vec<Vec<f32>> {
    fn to_dense_matrix(self) -> DenseMatrix<f32> {
        DenseMatrix::from_2d_vec(&self)
    }
}

impl IntoLabels for Vec<f32> {
    fn into_vec(self) -> Vec<f32> {
        self
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
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SupervisedModel {
    /// Settings for the model.
    settings: Settings,
    /// The training data.
    x_train: DenseMatrix<f32>,
    /// The training labels.
    y_train: Vec<f32>,
    /// The validation data.
    x_val: DenseMatrix<f32>,
    /// The validation labels.
    y_val: Vec<f32>,
    /// The number of classes in the data.
    number_of_classes: usize,
    /// The results of the model comparison.
    comparison: Vec<Model>,
    /// The final model.
    metamodel: Model,
    /// The preprocessing pipeline.
    preprocessing: (
        Option<PCA<f32, DenseMatrix<f32>>>,
        Option<SVD<f32, DenseMatrix<f32>>>,
    ),
}

impl SupervisedModel {
    /// Create a new supervised model. This function accepts various types of syntax. For instance, it will work for vectors:
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = automl::SupervisedModel::new(
    ///     (vec![vec![1.0; 5]; 5],
    ///     vec![1.0; 5]),
    ///     automl::Settings::default_regression(),
    /// );    
    /// ```
    /// It also works for some ndarray datatypes:
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// #[cfg(any(feature = "nd"))]
    /// let model = SupervisedModel::new(
    ///     (
    ///         ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]),
    ///         ndarray::arr1(&[1.0, 2.0])
    ///     ),
    ///     automl::Settings::default_regression(),
    /// );
    /// ```
    /// But you can also create a new supervised model from a [smartcore toy dataset](https://docs.rs/smartcore/0.2.0/smartcore/dataset/index.html)
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let model = SupervisedModel::new(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// ```
    /// You can even create a new supervised model directly from a CSV!
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// #[cfg(any(feature = "csv"))]
    /// let model = SupervisedModel::new(
    ///     ("data/diabetes.csv", 10),
    ///     Settings::default_regression()
    /// );
    /// ```
    /// And that CSV can even come from a URL
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// #[cfg(any(feature = "csv"))]
    /// let mut model = automl::SupervisedModel::new(
    ///         (
    ///         "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    ///         8,
    ///     ),
    ///     Settings::default_regression(),
    /// );
    pub fn new<D>(data: D, settings: Settings) -> Self
    where
        D: IntoSupervisedData,
    {
        let (x, y) = data.to_supervised_data();
        SupervisedModel::build(x, y, settings)
    }

    /// Load the supervised model from a file saved previously
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// # let mut model = SupervisedModel::new(
    /// #    smartcore::dataset::diabetes::load_dataset(),
    /// #    Settings::default_regression()
    /// # );
    /// # model.save("tests/load_that_model.aml");
    /// let model = SupervisedModel::new_from_file("tests/load_that_model.aml");
    /// # std::fs::remove_file("tests/load_that_model.aml");
    /// ```
    pub fn new_from_file(file_name: &str) -> Self {
        let mut buf: Vec<u8> = Vec::new();
        std::fs::File::open(file_name)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Cannot load model from file.");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    }

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
    pub fn predict<X>(&self, x: X) -> Vec<f32>
    where
        X: IntoFeatures,
    {
        let x = &self.preprocess(x.to_dense_matrix());
        match self.settings.final_model_approach {
            FinalModel::None => panic!(""),
            FinalModel::Best => self.predict_by_model(x, &self.comparison[0]),
            FinalModel::Blending { algorithm, .. } => self.predict_blended_model(x, algorithm),
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
            self.train_pca(self.x_train.clone(), number_of_components);
        }
        if let PreProcessing::ReplaceWithSVD {
            number_of_components,
        } = self.settings.preprocessing
        {
            self.train_svd(self.x_train.clone(), number_of_components);
        }

        // Preprocess the data
        self.x_train = self.preprocess(self.x_train.clone());

        // Split validatino out if blending
        if let FinalModel::Blending {
            meta_training_fraction,
            meta_testing_fraction: _,
            algorithm: _,
        } = &self.settings.final_model_approach
        {
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
            && std::mem::discriminant(&self.settings.preprocessing)
                != std::mem::discriminant(&PreProcessing::ReplaceWithPCA {
                    number_of_components: 1,
                })
            && std::mem::discriminant(&self.settings.preprocessing)
                != std::mem::discriminant(&PreProcessing::ReplaceWithSVD {
                    number_of_components: 1,
                })
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

        if let FinalModel::Blending {
            algorithm,
            meta_training_fraction,
            meta_testing_fraction,
        } = self.settings.final_model_approach
        {
            self.train_blended_model(algorithm, meta_training_fraction, meta_testing_fraction)
        }
    }

    /// Save the supervised model to a file for later use
    /// ```
    /// # use automl::{SupervisedModel, Settings};
    /// let mut model = SupervisedModel::new(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// );
    /// model.save("tests/save_that_model.aml");
    /// # std::fs::remove_file("tests/save_that_model.aml");
    /// ```
    pub fn save(&self, file_name: &str) {
        let serial = bincode::serialize(&self).expect("Cannot serialize model.");
        std::fs::File::create(file_name)
            .and_then(|mut f| f.write_all(&serial))
            .expect("Cannot write model to file.");
    }

    /// Save the best model for later use as a smartcore native object.
    /// ```
    /// # use automl::{SupervisedModel, Settings, settings::Algorithm};
    /// use std::io::Read;
    ///
    /// let mut model = SupervisedModel::new(
    ///     smartcore::dataset::diabetes::load_dataset(),
    ///     Settings::default_regression()
    /// # .only(Algorithm::Linear)
    /// );
    /// model.train();
    /// model.save("tests/save_best.sc");
    /// # std::fs::remove_file("tests/save_best.sc");
    /// ```
    pub fn save_best(&self, file_name: &str) {
        if let FinalModel::Best = self.settings.final_model_approach {
            std::fs::File::create(file_name)
                .and_then(|mut f| f.write_all(&self.comparison[0].model))
                .expect("Cannot write model to file.");
        }
    }
}

/// Private functions go here
impl SupervisedModel {
    /// Build a new supervised model
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `y` - The output data
    /// * `settings` - The settings for the model
    fn build(x: DenseMatrix<f32>, y: Vec<f32>, settings: Settings) -> Self {
        Self {
            settings,
            x_train: x,
            number_of_classes: Self::count_classes(&y),
            y_train: y,
            x_val: DenseMatrix::new(0, 0, vec![]),
            y_val: vec![],
            comparison: vec![],
            preprocessing: (None, None),
            metamodel: Default::default(),
        }
    }

    /// Train the supervised model.
    ///
    /// # Arguments
    ///
    /// * `algo` - The algorithm to use
    /// * `training_fraction` - The fraction of the data to use for training
    /// * `testing_fraction` - The fraction of the data to use for testing
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
        let model = algo.get_trainer()(&x_train, &y_train, &self.settings);

        // Score the model
        let train_score = self.settings.get_metric()(
            &y_train,
            &algo.get_predictor()(&x_train, &model, &self.settings),
            // &LassoRegressorWrapper::predict(&x_train, &model, &self.settings),
        );
        let test_score = self.settings.get_metric()(
            &y_test,
            &algo.get_predictor()(&x_test, &model, &self.settings),
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

    /// Predict using all of the trained models.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `algo` - The algorithm to use
    ///
    /// # Returns
    ///
    /// * The predicted values
    fn predict_blended_model(&self, x: &DenseMatrix<f32>, algo: Algorithm) -> Vec<f32> {
        // Make the data
        let mut meta_x: Vec<Vec<f32>> = Vec::new();
        for i in 0..self.comparison.len() {
            let model = &self.comparison[i];
            meta_x.push(self.predict_by_model(x, model))
        }

        //
        let xdm = DenseMatrix::from_2d_vec(&meta_x).transpose();
        let metamodel = &self.metamodel.model;

        // Train the model
        algo.get_predictor()(&xdm, metamodel, &self.settings)
    }

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
    fn predict_by_model(&self, x: &DenseMatrix<f32>, model: &Model) -> Vec<f32> {
        model.name.get_predictor()(x, &model.model, &self.settings)
    }

    /// Get interaction features for the data.
    ///
    /// # Arguments
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
    fn polynomial_features(mut x: DenseMatrix<f32>, order: usize) -> DenseMatrix<f32> {
        let (height, width) = x.shape();
        for n in 2..=order {
            let combinations = (0..width).combinations_with_replacement(n);
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

    /// Train PCA on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    fn train_pca(&mut self, x: DenseMatrix<f32>, n: usize) {
        let pca = PCA::fit(
            &x,
            PCAParameters::default()
                .with_n_components(n)
                .with_use_correlation_matrix(true),
        )
        .unwrap();
        self.preprocessing.0 = Some(pca);
    }

    /// Get PCA features for the data using the trained PCA preprocessor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    fn pca_features(&self, x: DenseMatrix<f32>, _: usize) -> DenseMatrix<f32> {
        self.preprocessing
            .0
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    /// Train SVD on the data for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `n` - The number of components to use
    fn train_svd(&mut self, x: DenseMatrix<f32>, n: usize) {
        let svd = SVD::fit(&x, SVDParameters::default().with_n_components(n)).unwrap();
        self.preprocessing.1 = Some(svd);
    }

    /// Get SVD features for the data.
    fn svd_features(&self, x: DenseMatrix<f32>, _: usize) -> DenseMatrix<f32> {
        self.preprocessing
            .1
            .as_ref()
            .unwrap()
            .transform(&x)
            .unwrap()
    }

    /// Preprocess the data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    ///
    /// # Returns
    ///
    /// * The preprocessed data
    fn preprocess(&self, x: DenseMatrix<f32>) -> DenseMatrix<f32> {
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

    /// Count the number of classes in the data.
    ///
    /// # Arguments
    ///
    /// * `y` - The data to count the classes in
    ///
    /// # Returns
    ///
    /// * The number of classes
    fn count_classes(y: &[f32]) -> usize {
        let mut sorted_targets = y.to_vec();
        sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
        sorted_targets.dedup();
        sorted_targets.len()
    }

    /// Record a model in the comparison.
    fn record_model(&mut self, model: (CrossValidationResult<f32>, Algorithm, Duration, Vec<u8>)) {
        self.comparison.push(Model {
            score: model.0,
            name: model.1,
            duration: model.2,
            model: model.3,
        });
        self.sort();
    }

    /// Sort the models in the comparison by their mean test scores.
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

/// This contains the results of a single model
#[derive(serde::Serialize, serde::Deserialize)]
struct Model {
    /// The cross validation score of the model
    #[serde(with = "CrossValidationResultDef")]
    score: CrossValidationResult<f32>,
    /// The algorithm used
    name: Algorithm,
    /// The time it took to train the model
    duration: Duration,
    /// What is this? TODO
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

/// This is a wrapper for the CrossValidationResult
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "CrossValidationResult::<f32>")]
struct CrossValidationResultDef {
    /// Vector with test scores on each cv split
    pub test_score: Vec<f32>,
    /// Vector with training scores on each cv split
    pub train_score: Vec<f32>,
}
