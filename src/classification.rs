//! Auto-ML for regression models

use super::traits::Classifier;
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

/// This contains the results of a single model, including the model itself
pub struct Model {
    model: Vec<u8>,
    accuracy: f32,
    name: String,
}

/// An enum for sorting
#[non_exhaustive]
pub enum SortBy {
    /// Sort by accuracy
    Accuracy,
}

/// The settings artifact for all classifications
pub struct Settings {
    sort_by: SortBy,
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
            sort_by: SortBy::Accuracy,
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
    pub fn sorted_by(mut self, sort_by: SortBy) -> Self {
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

/// This is the output from a model comparison operation
pub struct ComparisonResults {
    results: Vec<Model>,
    sort_by: SortBy,
}

impl ComparisonResults {
    /// Uses the best model to make a prediction
    pub fn predict_with_best_model(&self, x: &DenseMatrix<f32>) -> Vec<f32> {
        match self.results[0].name.as_str() {
            "Logistic Regression Classifier" => {
                let model: LogisticRegression<f32, DenseMatrix<f32>> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Random Forest Classifier" => {
                let model: RandomForestClassifier<f32> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Decision Tree Classifier" => {
                let model: DecisionTreeClassifier<f32> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "KNN Classifier" => {
                let model: KNNClassifier<f32, Euclidian> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            "Support Vector Classifier" => {
                let model: SVC<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(&*self.results[0].model).unwrap();
                model.predict(x).unwrap()
            }
            &_ => panic!("Unable to predict"),
        }
    }

    /// Returns a serialized version of the best model
    pub fn get_best_model(&self) -> Vec<u8> {
        self.results[0].model.clone()
    }

    fn add_model(&mut self, name: String, y_test: &Vec<f32>, y_pred: &Vec<f32>, model: Vec<u8>) {
        self.results.push(Model {
            model,
            accuracy: Accuracy {}.get_score(y_test, y_pred),
            name,
        });
        self.sort()
    }

    fn sort(&mut self) {
        match self.sort_by {
            SortBy::Accuracy => {
                self.results
                    .sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap_or(Equal));
            }
        }
    }

    fn new(sort_by: SortBy) -> Self {
        Self {
            results: Vec::new(),
            sort_by,
        }
    }
}

impl Display for ComparisonResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "Accuracy"]);
        for model_results in &self.results {
            table.add_row(vec![
                format!("{}", &model_results.name),
                format!("{}", model_results.accuracy),
            ]);
        }
        write!(f, "{}\n", table)
    }
}

/// This function compares all of the classification models available in the package.
/// ```
/// let data = smartcore::dataset::iris::load_dataset();
/// let settings = automl::classification::Settings::default();
/// let x = automl::classification::compare_models(data, settings);
/// print!("{}", x);
/// ```
pub fn compare_models(dataset: Dataset<f32, f32>, settings: Settings) -> ComparisonResults {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let mut sorted_targets = y.clone();
    sorted_targets.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Equal));
    sorted_targets.dedup();
    let number_of_classes = sorted_targets.len();

    let (x_test, x_train, y_test, y_train) =
        train_test_split(&x, &y, settings.testing_fraction, settings.shuffle);

    let mut results = ComparisonResults::new(settings.sort_by);

    // Do the standard linear model
    let model = LogisticRegression::fit(&x_train, &y_train, settings.logistic_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    // Do the standard linear model
    let model =
        RandomForestClassifier::fit(&x_train, &y_train, settings.random_forest_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    // Do the standard linear model
    let model = KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default()).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    let model =
        DecisionTreeClassifier::fit(&x_train, &y_train, settings.decision_tree_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let serial_model = bincode::serialize(&model).unwrap();
    results.add_model(model.name(), &y_test, &y_pred, serial_model);

    if number_of_classes == 2 {
        let model = SVC::fit(&x_train, &y_train, SVCParameters::default()).unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        let serial_model = bincode::serialize(&model).unwrap();
        results.add_model(model.name(), &y_test, &y_pred, serial_model);
    }

    results
}
