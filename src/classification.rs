//! Auto-ML for regression models

use comfy_table::{modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Table};
use smartcore::{
    dataset::Dataset,
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::{naive::dense_matrix::DenseMatrix, Matrix},
    linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters},
    math::{distance::Distance, num::RealNumber},
    metrics::accuracy,
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

trait Classifier {}
impl<T: RealNumber, M: Matrix<T>> Classifier for LogisticRegression<T, M> {}
impl<T: RealNumber> Classifier for RandomForestClassifier<T> {}
impl<T: RealNumber, D: Distance<Vec<T>, T>> Classifier for KNNClassifier<T, D> {}
impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Classifier for SVC<T, M, K> {}
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
/// let x = automl::classification::compare_models(data);
/// print!("{}", x);
/// ```
pub fn compare_models(dataset: Dataset<f32, f32>) -> ModelComparison {
    let x = DenseMatrix::from_array(dataset.num_samples, dataset.num_features, &dataset.data);
    // These are our target values
    let y = dataset.target;

    let mut results = Vec::new();

    // Do the standard linear model
    let model = LogisticRegression::fit(&x, &y, LogisticRegressionParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y, &y_pred),
        name: "Logistic Regression".to_string(),
    });

    // Do the standard linear model
    let model =
        RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y, &y_pred),
        name: "Random Forest Classifier".to_string(),
    });

    // Do the standard linear model
    let model = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y, &y_pred),
        name: "KNN Classifier".to_string(),
    });

    // let model = SVC::fit(&x, &y, SVCParameters::default()).unwrap();
    // let y_pred = model.predict(&x).unwrap();
    // results.push(ModelResult {
    //     model: Box::new(model),
    //     accuracy: accuracy(&y, &y_pred),
    //     name: "Support Vector Classifier".to_string(),
    // });

    let model =
        DecisionTreeClassifier::fit(&x, &y, DecisionTreeClassifierParameters::default()).unwrap();
    let y_pred = model.predict(&x).unwrap();
    results.push(ModelResult {
        model: Box::new(model),
        accuracy: accuracy(&y, &y_pred),
        name: "Support Vector Classifier".to_string(),
    });

    results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap_or(Equal));

    ModelComparison(results)
}
