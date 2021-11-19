//! Auto-ML for regression models

use crate::utils::Status;
use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};
use polars::prelude::{CsvReader, DataFrame, DataType, Float32Type, SerReader};

use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters},
    math::distance::euclidian::Euclidian,
    metrics::accuracy,
    model_selection::{cross_validate, CrossValidationResult, KFold},
    naive_bayes::{
        categorical::{CategoricalNB, CategoricalNBParameters},
        gaussian::{GaussianNB, GaussianNBParameters},
    },
    neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters},
    svm::{
        svc::{SVCParameters, SVC},
        LinearKernel,
    },
    tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters},
};
use std::{
    cmp::Ordering::Equal,
    fmt::{Display, Formatter},
};

/// An enum for sorting
#[non_exhaustive]
#[derive(PartialEq)]
pub enum Metric {
    /// Sort by accuracy
    Accuracy,
}

impl Display for Metric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::Accuracy => write!(f, "Accuracy"),
        }
    }
}

/// An enum containing possible  classification algorithms
#[derive(PartialEq)]
pub enum Algorithm {
    /// Decision tree classifier
    DecisionTree,
    /// KNN classifier
    KNN,
    /// Random forest classifier
    RandomForest,
    /// Support vector classifier
    SVC,
    /// Logistic regression classifier
    LogisticRegression,
    /// Gaussian Naive Bayes classifier
    GaussianNaiveBayes,
    /// Categorical Naive Bayes classifier
    CategoricalNaiveBayes,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::DecisionTree => write!(f, "Decision Tree Classifier"),
            Algorithm::KNN => write!(f, "KNN Classifier"),
            Algorithm::RandomForest => write!(f, "Random Forest Classifier"),
            Algorithm::LogisticRegression => write!(f, "Logistic Regression Classifier"),
            Algorithm::SVC => write!(f, "Support Vector Classifier"),
            Algorithm::GaussianNaiveBayes => write!(f, "Gaussian Naive Bayes"),
            Algorithm::CategoricalNaiveBayes => write!(f, "Categorical Naive Bayes"),
        }
    }
}

/// This is the output from a model comparison operation
pub struct Classifier {
    settings: Settings,
    x: DenseMatrix<f32>,
    y: Vec<f32>,
    comparison: Vec<Model>,
    final_model: Vec<u8>,
    number_of_classes: usize,
    status: Status,
}

impl Classifier {
    /// Establish a new classifier with settings
    pub fn new(x: DenseMatrix<f32>, y: Vec<f32>, settings: Settings) -> Self {
        Self {
            settings,
            x,
            y,
            comparison: vec![],
            final_model: vec![],
            number_of_classes: 0,
            status: Status::DataLoaded,
        }
    }

    /// Add settings to the classifier
    pub fn with_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    /// Runs a model comparison and trains a final model. [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(&mut self) {
        self.compare_models();
        self.train_final_model();
    }

    /// Add data to regressor object
    pub fn with_data(&mut self, x: DenseMatrix<f32>, y: Vec<f32>) {
        self.x = x;
        self.y = y;
        self.count_classes();
        self.status = Status::DataLoaded;
    }

    /// Add a dataset to regressor object
    pub fn with_dataset(&mut self, dataset: Dataset<f32, f32>) {
        self.x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        self.y = dataset.target;
        self.count_classes();
        self.status = Status::DataLoaded;
    }

    /// Add data from a csv
    pub fn with_data_from_csv(&mut self, filepath: &str, target: usize, header: bool) {
        let df = CsvReader::from_path(filepath)
            .unwrap()
            .infer_schema(None)
            .has_header(header)
            .finish()
            .unwrap();

        // Get target variables
        let target_column_name = df.get_column_names()[target];
        let series = df.column(target_column_name).unwrap().clone();
        let target_df = DataFrame::new(vec![series]).unwrap();
        let ndarray = target_df.to_ndarray::<Float32Type>().unwrap();
        self.y = ndarray.into_raw_vec();

        // Get the rest of the data
        let features = df.drop(target_column_name).unwrap();
        let (height, width) = features.shape();
        let ndarray = features.to_ndarray::<Float32Type>().unwrap();
        self.x = DenseMatrix::from_array(height, width, ndarray.as_slice().unwrap());

        // Set status
        self.status = Status::DataLoaded;
    }

    /// This function compares all of the classification models available in the package.
    pub fn compare_models(&mut self) {
        if self.status == Status::DataLoaded {
            if !self
                .settings
                .skiplist
                .contains(&Algorithm::LogisticRegression)
            {
                self.add_model(
                    Algorithm::LogisticRegression,
                    cross_validate(
                        LogisticRegression::fit,
                        &self.x,
                        &self.y,
                        self.settings.logistic_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::RandomForest) {
                self.add_model(
                    Algorithm::RandomForest,
                    cross_validate(
                        RandomForestClassifier::fit,
                        &self.x,
                        &self.y,
                        self.settings.random_forest_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::KNN) {
                self.add_model(
                    Algorithm::KNN,
                    cross_validate(
                        KNNClassifier::fit,
                        &self.x,
                        &self.y,
                        self.settings.knn_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self.settings.skiplist.contains(&Algorithm::DecisionTree) {
                self.add_model(
                    Algorithm::DecisionTree,
                    cross_validate(
                        DecisionTreeClassifier::fit,
                        &self.x,
                        &self.y,
                        self.settings.decision_tree_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self
                .settings
                .skiplist
                .contains(&Algorithm::GaussianNaiveBayes)
            {
                self.add_model(
                    Algorithm::GaussianNaiveBayes,
                    cross_validate(
                        GaussianNB::fit,
                        &self.x,
                        &self.y,
                        self.settings.gaussian_nb_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if !self
                .settings
                .skiplist
                .contains(&Algorithm::CategoricalNaiveBayes)
            {
                self.add_model(
                    Algorithm::CategoricalNaiveBayes,
                    cross_validate(
                        CategoricalNB::fit,
                        &self.x,
                        &self.y,
                        self.settings.categorical_nb_settings.clone(),
                        KFold::default().with_n_splits(self.settings.number_of_folds),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            if self.number_of_classes == 2 && !self.settings.skiplist.contains(&Algorithm::SVC) {
                self.add_model(
                    Algorithm::SVC,
                    cross_validate(
                        SVC::fit,
                        &self.x,
                        &self.y,
                        self.settings.svc_settings.clone(),
                        KFold::default().with_n_splits(3),
                        match self.settings.sort_by {
                            Metric::Accuracy => accuracy,
                        },
                    )
                    .unwrap(),
                );
            }

            self.status = Status::ModelsCompared;
        } else {
            panic!("You must load data before trying to compare models.")
        }
    }

    /// Trains the best model found during comparison
    pub fn train_final_model(&mut self) {
        match self.comparison[0].name {
            Algorithm::LogisticRegression => {
                self.final_model = bincode::serialize(
                    &LogisticRegression::fit(
                        &self.x,
                        &self.y,
                        self.settings.logistic_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::KNN => {
                self.final_model = bincode::serialize(
                    &KNNClassifier::fit(&self.x, &self.y, self.settings.knn_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }
            Algorithm::RandomForest => {
                self.final_model = bincode::serialize(
                    &RandomForestClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings.random_forest_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::DecisionTree => {
                self.final_model = bincode::serialize(
                    &DecisionTreeClassifier::fit(
                        &self.x,
                        &self.y,
                        self.settings.decision_tree_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
            Algorithm::SVC => {
                self.final_model = bincode::serialize(
                    &SVC::fit(&self.x, &self.y, self.settings.svc_settings.clone()).unwrap(),
                )
                .unwrap()
            }

            Algorithm::GaussianNaiveBayes => {
                self.final_model = bincode::serialize(
                    &GaussianNB::fit(&self.x, &self.y, self.settings.gaussian_nb_settings.clone())
                        .unwrap(),
                )
                .unwrap()
            }

            Algorithm::CategoricalNaiveBayes => {
                self.final_model = bincode::serialize(
                    &CategoricalNB::fit(
                        &self.x,
                        &self.y,
                        self.settings.categorical_nb_settings.clone(),
                    )
                    .unwrap(),
                )
                .unwrap()
            }
        }
        self.status = Status::FinalModelTrained;
    }

    /// Predict values using the best model
    pub fn predict(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.comparison[0].name {
            Algorithm::LogisticRegression => {
                let model: LogisticRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::RandomForest => {
                let model: RandomForestClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::DecisionTree => {
                let model: DecisionTreeClassifier<f32> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::KNN => {
                let model: KNNClassifier<f32, Euclidian> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::SVC => {
                let model: SVC<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::GaussianNaiveBayes => {
                let model: GaussianNB<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
            Algorithm::CategoricalNaiveBayes => {
                let model: CategoricalNB<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}

/// Private functions go here
impl Classifier {
    fn add_model(&mut self, name: Algorithm, score: CrossValidationResult<f32>) {
        self.comparison.push(Model { score, name });
        self.sort()
    }

    fn sort(&mut self) {
        self.comparison.sort_by(|a, b| {
            b.score
                .mean_test_score()
                .partial_cmp(&a.score.mean_test_score())
                .unwrap_or(Equal)
        });
    }

    fn count_classes(&mut self) {
        let mut sorted_targets = self.y.clone();
        sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
        sorted_targets.dedup();
        self.number_of_classes = sorted_targets.len();
    }
}

impl Display for Classifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec![
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new(format!("Training {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
            Cell::new(format!("Testing {}", self.settings.sort_by)).add_attribute(Attribute::Bold),
        ]);
        for model in &self.comparison {
            let mut row_vec = vec![];
            row_vec.push(format!("{}", &model.name));
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
        write!(f, "{}\n", table)
    }
}

impl Default for Classifier {
    fn default() -> Self {
        Self {
            settings: Default::default(),
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: vec![],
            final_model: vec![],
            number_of_classes: 0,
            status: Status::Starting,
        }
    }
}

/// This contains the results of a single model
struct Model {
    score: CrossValidationResult<f32>,
    name: Algorithm,
}

/// The settings artifact for all classifications
pub struct Settings {
    skiplist: Vec<Algorithm>,
    sort_by: Metric,
    number_of_folds: usize,
    shuffle: bool,
    verbose: bool,
    logistic_settings: LogisticRegressionParameters,
    random_forest_settings: RandomForestClassifierParameters,
    knn_settings: KNNClassifierParameters<f32, Euclidian>,
    svc_settings: SVCParameters<f32, DenseMatrix<f32>, LinearKernel>,
    decision_tree_settings: DecisionTreeClassifierParameters,
    gaussian_nb_settings: GaussianNBParameters<f32>,
    categorical_nb_settings: CategoricalNBParameters<f32>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            skiplist: vec![],
            sort_by: Metric::Accuracy,
            shuffle: true,
            verbose: true,
            logistic_settings: LogisticRegressionParameters::default(),
            random_forest_settings: RandomForestClassifierParameters::default(),
            knn_settings: KNNClassifierParameters::default(),
            svc_settings: SVCParameters::default(),
            decision_tree_settings: DecisionTreeClassifierParameters::default(),
            gaussian_nb_settings: GaussianNBParameters::default(),
            categorical_nb_settings: CategoricalNBParameters::default(),
            number_of_folds: 10,
        }
    }
}

impl Settings {
    /// Specify number of folds for cross-validation
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    pub fn skip(mut self, skip: Algorithm) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Adds a specific sorting function to the settings
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for random_forest
    pub fn with_random_forest_settings(
        mut self,
        settings: RandomForestClassifierParameters,
    ) -> Self {
        self.random_forest_settings = settings;
        self
    }

    /// Specify settings for logistic regression
    pub fn with_logistic_settings(mut self, settings: LogisticRegressionParameters) -> Self {
        self.logistic_settings = settings;
        self
    }

    /// Specify settings for support vector classifier
    pub fn with_svc_settings(
        mut self,
        settings: SVCParameters<f32, DenseMatrix<f32>, LinearKernel>,
    ) -> Self {
        self.svc_settings = settings;
        self
    }

    /// Specify settings for logistic regression
    pub fn with_decision_tree_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_settings = settings;
        self
    }

    /// Specify settings for logistic regression
    pub fn with_knn_settings(mut self, settings: KNNClassifierParameters<f32, Euclidian>) -> Self {
        self.knn_settings = settings;
        self
    }

    /// Specify settings for Gaussian Naive Bayes
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters<f32>) -> Self {
        self.gaussian_nb_settings = settings;
        self
    }
}

impl Display for Settings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Prep new table
        let mut table = Table::new();

        // Get list of algorithms to skip
        let mut skiplist = String::new();
        if self.skiplist.len() == 0 {
            skiplist.push_str("None");
        } else {
            for algorithm_to_skip in &self.skiplist {
                skiplist.push_str(&*format!("{}\n", algorithm_to_skip));
            }
        }

        // Build out the table
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_SOLID_INNER_BORDERS)
            .set_header(vec![
                Cell::new("Settings").add_attribute(Attribute::Bold),
                Cell::new("Value").add_attribute(Attribute::Bold),
            ])
            .add_row(vec![Cell::new("General").add_attribute(Attribute::Italic)])
            .add_row(vec!["    Verbose", &*format!("{}", self.verbose)])
            .add_row(vec!["    Sorting Metric", &*format!("{}", self.sort_by)])
            .add_row(vec!["    Shuffle Data", &*format!("{}", self.shuffle)])
            .add_row(vec![
                "    Number of CV Folds",
                &*format!("{}", self.number_of_folds),
            ])
            .add_row(vec![
                "    Skipped Algorithms",
                &*format!("{}", &skiplist[0..skiplist.len() - 1]),
            ]);
        write!(f, "{}\n", table)
    }
}
