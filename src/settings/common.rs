use super::{FinalAlgorithm, Metric, PreprocessingPipeline, PreprocessingStep};
use smartcore::model_selection::KFold;
use std::mem;

/// Settings shared by supervised learning models.
pub struct SupervisedSettings {
    pub(crate) sort_by: Metric,
    pub(crate) number_of_folds: usize,
    pub(crate) shuffle: bool,
    pub(crate) verbose: bool,
    pub(crate) final_model_approach: FinalAlgorithm,
    pub(crate) preprocessing: PreprocessingPipeline,
}

impl Default for SupervisedSettings {
    fn default() -> Self {
        Self {
            sort_by: Metric::Accuracy,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            final_model_approach: FinalAlgorithm::Best,
            preprocessing: PreprocessingPipeline::new(),
        }
    }
}

mod serde_impls {
    use super::SupervisedSettings;
    use serde::de::{self, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    impl Serialize for SupervisedSettings {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut state = serializer.serialize_struct("SupervisedSettings", 6)?;
            state.serialize_field("sort_by", &self.sort_by)?;
            state.serialize_field("number_of_folds", &self.number_of_folds)?;
            state.serialize_field("shuffle", &self.shuffle)?;
            state.serialize_field("verbose", &self.verbose)?;
            state.serialize_field("final_model_approach", &self.final_model_approach)?;
            state.serialize_field("preprocessing", &self.preprocessing)?;
            state.end()
        }
    }

    #[allow(clippy::too_many_lines)]
    impl<'de> Deserialize<'de> for SupervisedSettings {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                SortBy,
                NumberOfFolds,
                Shuffle,
                Verbose,
                FinalModelApproach,
                Preprocessing,
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
                            formatter.write_str("a valid field name for SupervisedSettings")
                        }

                        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where
                            E: de::Error,
                        {
                            match value {
                                "sort_by" => Ok(Field::SortBy),
                                "number_of_folds" => Ok(Field::NumberOfFolds),
                                "shuffle" => Ok(Field::Shuffle),
                                "verbose" => Ok(Field::Verbose),
                                "final_model_approach" => Ok(Field::FinalModelApproach),
                                "preprocessing" => Ok(Field::Preprocessing),
                                other => Err(de::Error::unknown_field(other, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct SupervisedSettingsVisitor;

            #[allow(clippy::elidable_lifetime_names)]
            impl<'de> Visitor<'de> for SupervisedSettingsVisitor {
                type Value = SupervisedSettings;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a map describing SupervisedSettings")
                }

                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: MapAccess<'de>,
                {
                    let mut sort_by = None;
                    let mut number_of_folds = None;
                    let mut shuffle = None;
                    let mut verbose = None;
                    let mut final_model_approach = None;
                    let mut preprocessing = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::SortBy => {
                                if sort_by.is_some() {
                                    return Err(de::Error::duplicate_field("sort_by"));
                                }
                                sort_by = Some(map.next_value()?);
                            }
                            Field::NumberOfFolds => {
                                if number_of_folds.is_some() {
                                    return Err(de::Error::duplicate_field("number_of_folds"));
                                }
                                number_of_folds = Some(map.next_value()?);
                            }
                            Field::Shuffle => {
                                if shuffle.is_some() {
                                    return Err(de::Error::duplicate_field("shuffle"));
                                }
                                shuffle = Some(map.next_value()?);
                            }
                            Field::Verbose => {
                                if verbose.is_some() {
                                    return Err(de::Error::duplicate_field("verbose"));
                                }
                                verbose = Some(map.next_value()?);
                            }
                            Field::FinalModelApproach => {
                                if final_model_approach.is_some() {
                                    return Err(de::Error::duplicate_field("final_model_approach"));
                                }
                                final_model_approach = Some(map.next_value()?);
                            }
                            Field::Preprocessing => {
                                if preprocessing.is_some() {
                                    return Err(de::Error::duplicate_field("preprocessing"));
                                }
                                preprocessing = Some(map.next_value()?);
                            }
                        }
                    }

                    let mut settings = SupervisedSettings::default();
                    if let Some(value) = sort_by {
                        settings.sort_by = value;
                    }
                    if let Some(value) = number_of_folds {
                        settings.number_of_folds = value;
                    }
                    if let Some(value) = shuffle {
                        settings.shuffle = value;
                    }
                    if let Some(value) = verbose {
                        settings.verbose = value;
                    }
                    if let Some(value) = final_model_approach {
                        settings.final_model_approach = value;
                    }
                    if let Some(value) = preprocessing {
                        settings.preprocessing = value;
                    }
                    Ok(settings)
                }
            }

            const FIELDS: &[&str] = &[
                "sort_by",
                "number_of_folds",
                "shuffle",
                "verbose",
                "final_model_approach",
                "preprocessing",
            ];
            deserializer.deserialize_struct("SupervisedSettings", FIELDS, SupervisedSettingsVisitor)
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
    /// Specify an explicit preprocessing pipeline.
    pub fn with_preprocessing(mut self, pre: PreprocessingPipeline) -> Self {
        self.preprocessing = pre;
        self
    }

    /// Append a preprocessing step to the pipeline.
    #[must_use]
    pub fn add_step(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing.push_step(step);
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
    fn with_preprocessing(mut self, pre: PreprocessingPipeline) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).with_preprocessing(pre);
        self
    }

    /// Delegate builder for [`SupervisedSettings::add_step`].
    #[must_use]
    fn add_step(mut self, step: PreprocessingStep) -> Self
    where
        Self: Sized,
    {
        let settings = self.supervised_mut();
        *settings = mem::take(settings).add_step(step);
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
