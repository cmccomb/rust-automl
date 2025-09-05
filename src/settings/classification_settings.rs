use super::{
    DecisionTreeClassifierParameters, FinalAlgorithm, GaussianNBParameters, KNNParameters,
    LogisticRegressionParameters, Metric, PreProcessing, RandomForestClassifierParameters,
    SettingsError, SupervisedSettings, WithSupervisedSettings,
};
use crate::settings::macros::with_settings_methods;
use smartcore::linalg::basic::arrays::Array1;
use smartcore::metrics::accuracy;
use smartcore::model_selection::KFold;
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
    /// Optional settings for Gaussian naive Bayes classifier
    pub(crate) gaussian_nb_settings: Option<GaussianNBParameters>,
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
            gaussian_nb_settings: Some(GaussianNBParameters::default()),
        }
    }
}

impl ClassificationSettings {
    /// Retrieve the metric function for classification tasks.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsError`] if no metric is set or if the metric is unsupported.
    pub fn get_metric<OUTPUT, OutputArray>(
        &self,
    ) -> Result<fn(&OutputArray, &OutputArray) -> f64, SettingsError>
    where
        OUTPUT: Number + Ord,
        OutputArray: Array1<OUTPUT>,
    {
        match self.supervised.sort_by {
            Metric::Accuracy => Ok(accuracy),
            Metric::None => Err(SettingsError::MetricNotSet),
            m => Err(SettingsError::UnsupportedMetric(m)),
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

    /// Specify settings for Gaussian naive Bayes classifier
    #[must_use]
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters) -> Self {
        self.gaussian_nb_settings = Some(settings);
        self
    }

    /// Set the number of folds for cross-validation.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::ClassificationSettings;
    /// let settings = ClassificationSettings::default().with_number_of_folds(5);
    /// assert_eq!(settings.get_kfolds().n_splits, 5);
    /// ```
    #[must_use]
    pub fn with_number_of_folds(self, n: usize) -> Self {
        <Self as WithSupervisedSettings>::with_number_of_folds(self, n)
    }

    /// Enable or disable shuffling of training data.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::ClassificationSettings;
    /// let settings = ClassificationSettings::default().shuffle_data(true);
    /// assert!(settings.get_kfolds().shuffle);
    /// ```
    #[must_use]
    pub fn shuffle_data(self, shuffle: bool) -> Self {
        <Self as WithSupervisedSettings>::shuffle_data(self, shuffle)
    }

    /// Enable or disable verbose logging.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::ClassificationSettings;
    /// let settings = ClassificationSettings::default().verbose(true);
    /// ```
    #[must_use]
    pub fn verbose(self, verbose: bool) -> Self {
        <Self as WithSupervisedSettings>::verbose(self, verbose)
    }

    /// Specify preprocessing strategy.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::{ClassificationSettings, PreProcessing};
    /// let settings = ClassificationSettings::default()
    ///     .with_preprocessing(PreProcessing::AddInteractions);
    /// ```
    #[must_use]
    pub fn with_preprocessing(self, pre: PreProcessing) -> Self {
        <Self as WithSupervisedSettings>::with_preprocessing(self, pre)
    }

    /// Choose the strategy for the final model.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::{ClassificationSettings, FinalAlgorithm};
    /// let settings = ClassificationSettings::default()
    ///     .with_final_model(FinalAlgorithm::Best);
    /// ```
    #[must_use]
    pub fn with_final_model(self, approach: FinalAlgorithm) -> Self {
        <Self as WithSupervisedSettings>::with_final_model(self, approach)
    }

    /// Set the metric used for sorting model results.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::{ClassificationSettings, Metric};
    /// let settings = ClassificationSettings::default().sorted_by(Metric::Accuracy);
    /// ```
    #[must_use]
    pub fn sorted_by(self, sort_by: Metric) -> Self {
        <Self as WithSupervisedSettings>::sorted_by(self, sort_by)
    }

    /// Create a [`KFold`] configuration from these settings.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::ClassificationSettings;
    /// let folds = ClassificationSettings::default().get_kfolds();
    /// assert_eq!(folds.n_splits, 10);
    /// ```
    #[must_use]
    pub fn get_kfolds(&self) -> KFold {
        <Self as WithSupervisedSettings>::get_kfolds(self)
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
