use super::{FinalAlgorithm, Metric, PreProcessing};
use smartcore::model_selection::KFold;

/// Settings shared by supervised learning models.
pub struct SupervisedSettings {
    pub(crate) sort_by: Metric,
    pub(crate) number_of_folds: usize,
    pub(crate) shuffle: bool,
    pub(crate) verbose: bool,
    pub(crate) final_model_approach: FinalAlgorithm,
    pub(crate) preprocessing: PreProcessing,
}

impl Default for SupervisedSettings {
    fn default() -> Self {
        Self {
            sort_by: Metric::Accuracy,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            final_model_approach: FinalAlgorithm::Best,
            preprocessing: PreProcessing::None,
        }
    }
}

impl SupervisedSettings {
    pub(crate) fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.number_of_folds)
            .with_shuffle(self.shuffle)
    }

    #[must_use]
    /// Set the number of folds for cross-validation.
    pub const fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    #[must_use]
    /// Enable or disable shuffling of training data.
    pub const fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    #[must_use]
    /// Enable or disable verbose logging.
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    #[must_use]
    /// Specify preprocessing strategy.
    pub const fn with_preprocessing(mut self, pre: PreProcessing) -> Self {
        self.preprocessing = pre;
        self
    }

    #[must_use]
    /// Choose the strategy for the final model.
    pub fn with_final_model(mut self, approach: FinalAlgorithm) -> Self {
        self.final_model_approach = approach;
        self
    }

    #[must_use]
    /// Set the metric used for sorting model results.
    pub const fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }
}
