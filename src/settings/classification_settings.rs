use super::{
    CategoricalNBParameters, DecisionTreeClassifierParameters, FinalAlgorithm,
    GaussianNBParameters, KNNParameters, LogisticRegressionParameters, Metric,
    MultinomialNBParameters, PreProcessing, RandomForestClassifierParameters, SVCParameters,
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
    /// Optional settings for categorical naive Bayes classifier
    pub(crate) categorical_nb_settings: Option<CategoricalNBParameters>,
    /// Optional settings for multinomial naive Bayes classifier
    pub(crate) multinomial_nb_settings: Option<MultinomialNBParameters>,
    /// Optional settings for support vector classifier
    pub(crate) svc_settings: Option<SVCParameters>,
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
            categorical_nb_settings: None,
            multinomial_nb_settings: None,
            svc_settings: None,
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
        /// Specify settings for support vector classifier
        with_svc_settings, svc_settings, SVCParameters;
    }

    /// Specify settings for Gaussian naive Bayes classifier
    #[must_use]
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters) -> Self {
        self.gaussian_nb_settings = Some(settings);
        self
    }

    /// Specify settings for categorical naive Bayes classifier
    #[must_use]
    pub fn with_categorical_nb_settings(mut self, settings: CategoricalNBParameters) -> Self {
        self.categorical_nb_settings = Some(settings);
        self
    }

    /// Specify settings for multinomial naive Bayes classifier
    #[must_use]
    pub fn with_multinomial_nb_settings(mut self, settings: MultinomialNBParameters) -> Self {
        self.multinomial_nb_settings = Some(settings);
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

mod serde_impls {
    use super::{
        CategoricalNBParameters, ClassificationSettings, DecisionTreeClassifierParameters,
        GaussianNBParameters, KNNParameters, LogisticRegressionParameters, MultinomialNBParameters,
        RandomForestClassifierParameters, SVCParameters, SupervisedSettings,
    };
    use serde::de::{self, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    impl Serialize for ClassificationSettings {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut state = serializer.serialize_struct("ClassificationSettings", 9)?;
            state.serialize_field("supervised", &self.supervised)?;
            state.serialize_field("knn_classifier_settings", &self.knn_classifier_settings)?;
            state.serialize_field(
                "decision_tree_classifier_settings",
                &self.decision_tree_classifier_settings,
            )?;
            state.serialize_field(
                "random_forest_classifier_settings",
                &self.random_forest_classifier_settings,
            )?;
            state.serialize_field(
                "logistic_regression_settings",
                &self.logistic_regression_settings,
            )?;
            state.serialize_field("gaussian_nb_settings", &self.gaussian_nb_settings)?;
            state.serialize_field("categorical_nb_settings", &self.categorical_nb_settings)?;
            state.serialize_field("multinomial_nb_settings", &self.multinomial_nb_settings)?;
            state.serialize_field("svc_settings", &self.svc_settings)?;
            state.end()
        }
    }

    #[allow(clippy::too_many_lines)]
    impl<'de> Deserialize<'de> for ClassificationSettings {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                Supervised,
                KnnClassifierSettings,
                DecisionTreeClassifierSettings,
                RandomForestClassifierSettings,
                LogisticRegressionSettings,
                GaussianNbSettings,
                CategoricalNbSettings,
                MultinomialNbSettings,
                SvcSettings,
            }

            impl<'de> Deserialize<'de> for Field {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    struct FieldVisitor;

                    #[allow(clippy::elidable_lifetime_names)]
                    impl<'de> Visitor<'de> for FieldVisitor {
                        type Value = Field;

                        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                            formatter.write_str("a valid field name for ClassificationSettings")
                        }

                        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where
                            E: de::Error,
                        {
                            match value {
                                "supervised" => Ok(Field::Supervised),
                                "knn_classifier_settings" => Ok(Field::KnnClassifierSettings),
                                "decision_tree_classifier_settings" => {
                                    Ok(Field::DecisionTreeClassifierSettings)
                                }
                                "random_forest_classifier_settings" => {
                                    Ok(Field::RandomForestClassifierSettings)
                                }
                                "logistic_regression_settings" => {
                                    Ok(Field::LogisticRegressionSettings)
                                }
                                "gaussian_nb_settings" => Ok(Field::GaussianNbSettings),
                                "categorical_nb_settings" => Ok(Field::CategoricalNbSettings),
                                "multinomial_nb_settings" => Ok(Field::MultinomialNbSettings),
                                "svc_settings" => Ok(Field::SvcSettings),
                                other => Err(de::Error::unknown_field(other, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct ClassificationSettingsVisitor;

            #[allow(clippy::elidable_lifetime_names)]
            impl<'de> Visitor<'de> for ClassificationSettingsVisitor {
                type Value = ClassificationSettings;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a map describing ClassificationSettings")
                }

                #[allow(clippy::too_many_lines)]
                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: MapAccess<'de>,
                {
                    let mut supervised: Option<SupervisedSettings> = None;
                    let mut knn_classifier_settings: Option<Option<KNNParameters>> = None;
                    let mut decision_tree_classifier_settings: Option<
                        Option<DecisionTreeClassifierParameters>,
                    > = None;
                    let mut random_forest_classifier_settings: Option<
                        Option<RandomForestClassifierParameters>,
                    > = None;
                    let mut logistic_regression_settings: Option<
                        Option<LogisticRegressionParameters<f64>>,
                    > = None;
                    let mut gaussian_nb_settings: Option<Option<GaussianNBParameters>> = None;
                    let mut categorical_nb_settings: Option<Option<CategoricalNBParameters>> = None;
                    let mut multinomial_nb_settings: Option<Option<MultinomialNBParameters>> = None;
                    let mut svc_settings: Option<Option<SVCParameters>> = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::Supervised => {
                                if supervised.is_some() {
                                    return Err(de::Error::duplicate_field("supervised"));
                                }
                                supervised = Some(map.next_value()?);
                            }
                            Field::KnnClassifierSettings => {
                                if knn_classifier_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "knn_classifier_settings",
                                    ));
                                }
                                knn_classifier_settings = Some(map.next_value()?);
                            }
                            Field::DecisionTreeClassifierSettings => {
                                if decision_tree_classifier_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "decision_tree_classifier_settings",
                                    ));
                                }
                                decision_tree_classifier_settings = Some(map.next_value()?);
                            }
                            Field::RandomForestClassifierSettings => {
                                if random_forest_classifier_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "random_forest_classifier_settings",
                                    ));
                                }
                                random_forest_classifier_settings = Some(map.next_value()?);
                            }
                            Field::LogisticRegressionSettings => {
                                if logistic_regression_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "logistic_regression_settings",
                                    ));
                                }
                                logistic_regression_settings = Some(map.next_value()?);
                            }
                            Field::GaussianNbSettings => {
                                if gaussian_nb_settings.is_some() {
                                    return Err(de::Error::duplicate_field("gaussian_nb_settings"));
                                }
                                gaussian_nb_settings = Some(map.next_value()?);
                            }
                            Field::CategoricalNbSettings => {
                                if categorical_nb_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "categorical_nb_settings",
                                    ));
                                }
                                categorical_nb_settings = Some(map.next_value()?);
                            }
                            Field::MultinomialNbSettings => {
                                if multinomial_nb_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "multinomial_nb_settings",
                                    ));
                                }
                                multinomial_nb_settings = Some(map.next_value()?);
                            }
                            Field::SvcSettings => {
                                if svc_settings.is_some() {
                                    return Err(de::Error::duplicate_field("svc_settings"));
                                }
                                svc_settings = Some(map.next_value()?);
                            }
                        }
                    }

                    let mut settings = ClassificationSettings::default();
                    if let Some(value) = supervised {
                        settings.supervised = value;
                    }
                    if let Some(value) = knn_classifier_settings {
                        settings.knn_classifier_settings = value;
                    }
                    if let Some(value) = decision_tree_classifier_settings {
                        settings.decision_tree_classifier_settings = value;
                    }
                    if let Some(value) = random_forest_classifier_settings {
                        settings.random_forest_classifier_settings = value;
                    }
                    if let Some(value) = logistic_regression_settings {
                        settings.logistic_regression_settings = value;
                    }
                    if let Some(value) = gaussian_nb_settings {
                        settings.gaussian_nb_settings = value;
                    }
                    if let Some(value) = categorical_nb_settings {
                        settings.categorical_nb_settings = value;
                    }
                    if let Some(value) = multinomial_nb_settings {
                        settings.multinomial_nb_settings = value;
                    }
                    if let Some(value) = svc_settings {
                        settings.svc_settings = value;
                    }
                    Ok(settings)
                }
            }

            const FIELDS: &[&str] = &[
                "supervised",
                "knn_classifier_settings",
                "decision_tree_classifier_settings",
                "random_forest_classifier_settings",
                "logistic_regression_settings",
                "gaussian_nb_settings",
                "categorical_nb_settings",
                "multinomial_nb_settings",
                "svc_settings",
            ];

            deserializer.deserialize_struct(
                "ClassificationSettings",
                FIELDS,
                ClassificationSettingsVisitor,
            )
        }
    }
}
