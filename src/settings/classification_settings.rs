use super::{
    DecisionTreeClassifierParameters, FinalAlgorithm, KNNClassifierParameters, Metric,
    PreProcessing, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::arrays::Array1;
use smartcore::numbers::basenum::Number;
use smartcore::{metrics::accuracy, model_selection::KFold};

/// Settings for classification models
pub struct ClassificationSettings {
    /// The metric to sort by
    pub(crate) sort_by: Metric,
    /// The number of folds for cross-validation
    pub(crate) number_of_folds: usize,
    /// Whether to shuffle the data
    pub(crate) shuffle: bool,
    /// Whether to be verbose
    pub(crate) verbose: bool,
    /// The approach to use for the final model
    pub(crate) final_model_approach: FinalAlgorithm,
    /// The kind of preprocessing to perform
    pub(crate) preprocessing: PreProcessing,
    /// Optional settings for KNN classifier
    pub(crate) knn_classifier_settings: Option<KNNClassifierParameters>,
    /// Optional settings for decision tree classifier
    pub(crate) decision_tree_classifier_settings: Option<DecisionTreeClassifierParameters>,
    /// Optional settings for random forest classifier
    pub(crate) random_forest_classifier_settings: Option<RandomForestClassifierParameters>,
}

impl Default for ClassificationSettings {
    fn default() -> Self {
        Self {
            sort_by: Metric::Accuracy,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            final_model_approach: FinalAlgorithm::Best,
            preprocessing: PreProcessing::None,
            knn_classifier_settings: Some(KNNClassifierParameters::default()),
            decision_tree_classifier_settings: Some(DecisionTreeClassifierParameters::default()),
            random_forest_classifier_settings: Some(RandomForestClassifierParameters::default()),
        }
    }
}

impl ClassificationSettings {
    /// Get the k-fold cross-validator
    pub(crate) fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.number_of_folds)
            .with_shuffle(self.shuffle)
    }

    pub(crate) fn get_metric<OUTPUT, OutputArray>(&self) -> fn(&OutputArray, &OutputArray) -> f64
    where
        OUTPUT: Number + Ord,
        OutputArray: Array1<OUTPUT>,
    {
        match self.sort_by {
            Metric::Accuracy => accuracy,
            Metric::None => panic!("A metric must be set."),
            _ => panic!("Unsupported metric for classification"),
        }
    }

    /// Specify number of folds for cross-validation
    #[must_use]
    pub const fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether data should be shuffled
    #[must_use]
    pub const fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify whether to be verbose
    #[must_use]
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Specify what type of preprocessing should be performed
    #[must_use]
    pub const fn with_preprocessing(mut self, pre: PreProcessing) -> Self {
        self.preprocessing = pre;
        self
    }

    /// Specify what type of final model to use
    #[must_use]
    pub fn with_final_model(mut self, approach: FinalAlgorithm) -> Self {
        self.final_model_approach = approach;
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
}
