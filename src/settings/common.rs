use super::{FinalAlgorithm, Metric, PreProcessing};
use smartcore::model_selection::KFold;
use std::mem;

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

/// Trait exposing builders that delegate to [`SupervisedSettings`].
pub trait WithSupervisedSettings {
    /// Immutable access to inner [`SupervisedSettings`].
    fn supervised(&self) -> &SupervisedSettings;

    /// Mutable access to inner [`SupervisedSettings`].
    fn supervised_mut(&mut self) -> &mut SupervisedSettings;

    /// Delegate builder for [`SupervisedSettings::with_number_of_folds`].
    #[must_use]
    fn with_number_of_folds(mut self, n: usize) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).with_number_of_folds(n);
        self
    }

    /// Delegate builder for [`SupervisedSettings::shuffle_data`].
    #[must_use]
    fn shuffle_data(mut self, shuffle: bool) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).shuffle_data(shuffle);
        self
    }

    /// Delegate builder for [`SupervisedSettings::verbose`].
    #[must_use]
    fn verbose(mut self, verbose: bool) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).verbose(verbose);
        self
    }

    /// Delegate builder for [`SupervisedSettings::with_preprocessing`].
    #[must_use]
    fn with_preprocessing(mut self, pre: PreProcessing) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).with_preprocessing(pre);
        self
    }

    /// Delegate builder for [`SupervisedSettings::with_final_model`].
    #[must_use]
    fn with_final_model(mut self, approach: FinalAlgorithm) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).with_final_model(approach);
        self
    }

    /// Delegate builder for [`SupervisedSettings::sorted_by`].
    #[must_use]
    fn sorted_by(mut self, sort_by: Metric) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).sorted_by(sort_by);
        self
    }

    /// Delegate to `SupervisedSettings::get_kfolds`.
    fn get_kfolds(&self) -> KFold {
        self.supervised().get_kfolds()
    }
}
