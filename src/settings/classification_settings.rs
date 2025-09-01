use super::{
    DecisionTreeClassifierParameters, FinalAlgorithm, KNNClassifierParameters,
    LogisticRegressionParameters, Metric, PreProcessing, RandomForestClassifierParameters,
    SupervisedSettings,
};
use smartcore::linalg::basic::arrays::Array1;
use smartcore::numbers::basenum::Number;
use smartcore::{metrics::accuracy, model_selection::KFold};

/// Settings for classification models
pub struct ClassificationSettings {
    /// Shared supervised settings
    pub(crate) supervised: SupervisedSettings,
    /// Optional settings for KNN classifier
    pub(crate) knn_classifier_settings: Option<KNNClassifierParameters>,
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
            knn_classifier_settings: Some(KNNClassifierParameters::default()),
            decision_tree_classifier_settings: Some(DecisionTreeClassifierParameters::default()),
            random_forest_classifier_settings: Some(RandomForestClassifierParameters::default()),
            logistic_regression_settings: Some(LogisticRegressionParameters::default()),
        }
    }
}

impl ClassificationSettings {
    /// Get the k-fold cross-validator
    pub(crate) fn get_kfolds(&self) -> KFold {
        self.supervised.get_kfolds()
    }

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

    /// Specify number of folds for cross-validation
    #[must_use]
    pub const fn with_number_of_folds(mut self, n: usize) -> Self {
        self.supervised = self.supervised.with_number_of_folds(n);
        self
    }

    /// Specify whether data should be shuffled
    #[must_use]
    pub const fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.supervised = self.supervised.shuffle_data(shuffle);
        self
    }

    /// Specify whether to be verbose
    #[must_use]
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.supervised = self.supervised.verbose(verbose);
        self
    }

    /// Specify what type of preprocessing should be performed
    #[must_use]
    pub const fn with_preprocessing(mut self, pre: PreProcessing) -> Self {
        self.supervised = self.supervised.with_preprocessing(pre);
        self
    }

    /// Specify what type of final model to use
    #[must_use]
    pub fn with_final_model(mut self, approach: FinalAlgorithm) -> Self {
        self.supervised = self.supervised.with_final_model(approach);
        self
    }

    /// Specify settings for KNN classifier
    #[must_use]
    pub const fn with_knn_classifier_settings(mut self, settings: KNNClassifierParameters) -> Self {
        self.knn_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for decision tree classifier
    #[must_use]
    pub const fn with_decision_tree_classifier_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for random forest classifier
    #[must_use]
    pub const fn with_random_forest_classifier_settings(
        mut self,
        settings: RandomForestClassifierParameters,
    ) -> Self {
        self.random_forest_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for logistic regression classifier
    #[must_use]
    pub const fn with_logistic_regression_settings(
        mut self,
        settings: LogisticRegressionParameters<f64>,
    ) -> Self {
        self.logistic_regression_settings = Some(settings);
        self
    }
}
