//! Auto-ML for regression models

use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters;
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::{naive::dense_matrix::DenseMatrix, Matrix},
    linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters},
    math::{distance::Distance, num::RealNumber},
    metrics::accuracy,
    model_selection::train_test_split,
    neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters},
    svm::{
        svc::{SVCParameters, SVC},
        Kernel,
    },
    tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters},
};
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};

/// This contains the results of a single model, including the model itself
pub struct ModelResult {
    model: Box<dyn Classifier>,
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
    // knn_settings: KNNClassifierParameters<T, D>,
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
            // knn_settings: KNNClassifierParameters::default(),
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

    /// Specify settings for logistic regression
    pub fn with_decision_tree_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_settings = settings;
        self
    }
}

trait Classifier {}
impl<T: RealNumber, M: Matrix<T>> Classifier for LogisticRegression<T, M> {}
impl<T: RealNumber> Classifier for RandomForestClassifier<T> {}
impl<T: RealNumber, D: Distance<Vec<T>, T>> Classifier for KNNClassifier<T, D> {}
impl<T: RealNumber> Classifier for DecisionTreeClassifier<T> {}

/// This is the output from a model comparison operation
pub struct ModelComparison(Vec<ModelResult>);

impl Display for ModelComparison {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.apply_modifier(UTF8_SOLID_INNER_BORDERS);
        table.set_header(vec!["Model", "Accuracy"]);
        for model_results in &self.0 {
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
pub fn compare_models(dataset: Dataset<f32, f32>, settings: Settings) -> ModelComparison {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let (x_test, x_train, y_test, y_train) =
        train_test_split(&x, &y, settings.testing_fraction, settings.shuffle);

    let mut results = Vec::new();

    // Do the standard linear model
    let model = LogisticRegression::fit(&x_train, &y_train, settings.logistic_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y_test, &y_pred),
        name: "Logistic Regression".to_string(),
    });

    // Do the standard linear model
    let model =
        RandomForestClassifier::fit(&x_train, &y_train, settings.random_forest_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y_test, &y_pred),
        name: "Random Forest Classifier".to_string(),
    });

    // Do the standard linear model
    let model = KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default()).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y_test, &y_pred),
        name: "KNN Classifier".to_string(),
    });

    let model =
        DecisionTreeClassifier::fit(&x_train, &y_train, settings.decision_tree_settings).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y_test, &y_pred),
        name: "Support Vector Classifier".to_string(),
    });

    match settings.sort_by {
        SortBy::Accuracy => {
            results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap_or(Equal));
        }
    }

    ModelComparison(results)
}
