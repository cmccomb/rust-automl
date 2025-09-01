use super::{
    DecisionTreeClassifierParameters, KNNParameters, LogisticRegressionParameters, Metric,
    RandomForestClassifierParameters, SupervisedSettings, WithSupervisedSettings,
};
use crate::settings::macros::with_settings_methods;
use smartcore::linalg::basic::arrays::Array1;
use smartcore::metrics::accuracy;
use smartcore::numbers::basenum::Number;

/// Settings for classification models
pub struct ClassificationSettings {
    /// Shared supervised settings
    pub(crate) supervised: SupervisedSettings,
    /// Optional settings for KNN classifier
    pub(crate) knn_classifier_settings: Option<KNNParameters>,
    /// Optional settings for decision tree classifier
    pub(crate) decision_tree_classifier_settings: Option<DecisionTreeClassifierParameters>,
    /// Optional settings for random forest classifier
    pub(crate) random_forest_classifier_settings: Option<RandomForestClassifierParameters>,
    /// Optional settings for logistic regression classifier
    pub(crate) logistic_regression_settings: Option<LogisticRegressionParameters<f64>>,
}

impl Default for ClassificationSettings {
    fn default() -> Self {
        Self {
            supervised: SupervisedSettings {
                sort_by: Metric::Accuracy,
                ..SupervisedSettings::default()
            },
            knn_classifier_settings: Some(KNNParameters::default()),
            decision_tree_classifier_settings: Some(DecisionTreeClassifierParameters::default()),
            random_forest_classifier_settings: Some(RandomForestClassifierParameters::default()),
            logistic_regression_settings: Some(LogisticRegressionParameters::default()),
        }
    }
}

impl ClassificationSettings {
    pub(crate) fn get_metric<OUTPUT, OutputArray>(&self) -> fn(&OutputArray, &OutputArray) -> f64
    where
        OUTPUT: Number + Ord,
        OutputArray: Array1<OUTPUT>,
    {
        match self.supervised.sort_by {
            Metric::Accuracy => accuracy,
            Metric::None => panic!("A metric must be set."),
            _ => panic!("Unsupported metric for classification"),
        }
    }

    with_settings_methods! {
        /// Specify settings for KNN classifier
        with_knn_classifier_settings, knn_classifier_settings, KNNParameters;
        /// Specify settings for decision tree classifier
        with_decision_tree_classifier_settings, decision_tree_classifier_settings, DecisionTreeClassifierParameters;
        /// Specify settings for random forest classifier
        with_random_forest_classifier_settings, random_forest_classifier_settings, RandomForestClassifierParameters;
        /// Specify settings for logistic regression classifier
        with_logistic_regression_settings, logistic_regression_settings, LogisticRegressionParameters<f64>;
    }
}

impl WithSupervisedSettings for ClassificationSettings {
    fn supervised(&self) -> &SupervisedSettings {
        &self.supervised
    }

    fn supervised_mut(&mut self) -> &mut SupervisedSettings {
        &mut self.supervised
    }
}
