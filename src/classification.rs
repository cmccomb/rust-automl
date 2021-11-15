//! Auto-ML for regression models

use super::traits::ValidClassifier;
use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters},
    math::distance::euclidian::Euclidian,
    metrics::accuracy::Accuracy,
    model_selection::train_test_split,
    neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters},
    svm::{
        svc::{SVCParameters, SVC},
        LinearKernel,
    },
    tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters},
};
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};

/// An enum for sorting
#[non_exhaustive]
pub enum Metric {
    /// Sort by accuracy
    Accuracy,
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
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::DecisionTree => write!(f, "Decision Tree Classifier"),
            Algorithm::KNN => write!(f, "KNN Classifier"),
            Algorithm::RandomForest => write!(f, "Random Forest Classifier"),
            Algorithm::LogisticRegression => write!(f, "Logistic Regression Classifier"),
            Algorithm::SVC => write!(f, "Support Vector Classifier"),
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
}

impl Classifier {
    /// [Zhu Li, do the thing!](https://www.youtube.com/watch?v=mofRHlO1E_A)
    pub fn auto(settings: Settings, x: DenseMatrix<f32>, y: Vec<f32>) -> Self {
        let mut classifier = Self::new(settings);
        classifier.with_data(x, y);
        classifier.compare_models();
        classifier.train_final_model();
        classifier
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
        }
    }

    /// Returns a serialized version of the best model
    pub fn get_best_model(&self) -> Vec<u8> {
        self.final_model.clone()
    }

    fn add_model(&mut self, name: Algorithm, y_test: &Vec<f32>, y_pred: &Vec<f32>) {
        self.comparison.push(Model {
            accuracy: Accuracy {}.get_score(y_test, y_pred),
            name,
        });
        self.sort()
    }

    fn sort(&mut self) {
        match self.settings.sort_by {
            Metric::Accuracy => {
                self.comparison
                    .sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap_or(Equal));
            }
        }
    }

    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            x: DenseMatrix::new(0, 0, vec![]),
            y: vec![],
            comparison: vec![],
            final_model: vec![],
            number_of_classes: 0,
        }
    }

    /// Add data to regressor object
    pub fn with_data(&mut self, x: DenseMatrix<f32>, y: Vec<f32>) {
        self.x = x;
        self.y = y;
        self.count_classes();
    }

    /// Add a dataset to regressor object
    pub fn with_dataset(&mut self, dataset: Dataset<f32, f32>) {
        self.x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
        self.y = dataset.target;
        self.count_classes();
    }

    fn count_classes(&mut self) {
        let mut sorted_targets = self.y.clone();
        sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
        sorted_targets.dedup();
        self.number_of_classes = sorted_targets.len();
    }

    /// This function compares all of the classification models available in the package.
    pub fn compare_models(&mut self) {
        let (x_test, x_train, y_test, y_train) = train_test_split(
            &self.x,
            &self.y,
            self.settings.testing_fraction,
            self.settings.shuffle,
        );
        // Do the standard linear model
        let model =
            LogisticRegression::fit(&x_train, &y_train, self.settings.logistic_settings.clone())
                .unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        self.add_model(Algorithm::LogisticRegression, &y_test, &y_pred);

        // Do the standard linear model
        let model = RandomForestClassifier::fit(
            &x_train,
            &y_train,
            self.settings.random_forest_settings.clone(),
        )
        .unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        self.add_model(Algorithm::RandomForest, &y_test, &y_pred);

        // Do the standard linear model
        let model =
            KNNClassifier::fit(&x_train, &y_train, self.settings.knn_settings.clone()).unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        self.add_model(Algorithm::KNN, &y_test, &y_pred);

        let model = DecisionTreeClassifier::fit(
            &x_train,
            &y_train,
            self.settings.decision_tree_settings.clone(),
        )
        .unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        self.add_model(Algorithm::DecisionTree, &y_test, &y_pred);

        if self.number_of_classes == 2 {
            let model = SVC::fit(&x_train, &y_train, self.settings.svc_settings.clone()).unwrap();
            let y_pred = model.predict(&x_test).unwrap();
            self.add_model(Algorithm::SVC, &y_test, &y_pred);
        }
    }
}

impl Display for Classifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "Accuracy"]);
        for model in &self.comparison {
            table.add_row(vec![
                format!("{}", &model.name),
                format!("{}", model.accuracy),
            ]);
        }
        write!(f, "{}\n", table)
    }
}

/// This contains the results of a single model
struct Model {
    accuracy: f32,
    name: Algorithm,
}

/// The settings artifact for all classifications
pub struct Settings {
    sort_by: Metric,
    testing_fraction: f32,
    shuffle: bool,
    logistic_settings: LogisticRegressionParameters,
    random_forest_settings: RandomForestClassifierParameters,
    knn_settings: KNNClassifierParameters<f32, Euclidian>,
    svc_settings: SVCParameters<f32, DenseMatrix<f32>, LinearKernel>,
    decision_tree_settings: DecisionTreeClassifierParameters,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: Metric::Accuracy,
            testing_fraction: 0.3,
            shuffle: true,
            logistic_settings: LogisticRegressionParameters::default(),
            random_forest_settings: RandomForestClassifierParameters::default(),
            knn_settings: KNNClassifierParameters::default(),
            svc_settings: SVCParameters::default(),
            decision_tree_settings: DecisionTreeClassifierParameters::default(),
        }
    }
}

impl Settings {
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
}
