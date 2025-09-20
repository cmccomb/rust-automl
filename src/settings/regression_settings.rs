//! Settings for regression models

#![allow(clippy::struct_field_names)]

use super::{
    DecisionTreeRegressorParameters, ElasticNetParameters, FinalAlgorithm, KNNParameters,
    LassoParameters, LinearRegressionParameters, Metric, PreProcessing,
    RandomForestRegressorParameters, RidgeRegressionParameters, SVRParameters, SettingsError,
    SupervisedSettings, WithSupervisedSettings, XGRegressorParameters,
};
use crate::algorithms::RegressionAlgorithm;
use crate::settings::macros::with_settings_methods;

use smartcore::linalg::basic::arrays::Array1;
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::metrics::{mean_absolute_error, mean_squared_error, r2};
use smartcore::model_selection::KFold;
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};
use std::fmt::{Display, Formatter};

/// Settings for regression models.
///
/// Any algorithms in the `skiplist` member will be skipped during training.
pub struct RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + 'static,
    OUTPUT: FloatNumber + 'static,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>
        + 'static,
    OutputArray: Array1<OUTPUT> + 'static,
{
    /// Shared supervised settings
    pub(crate) supervised: SupervisedSettings,
    /// The algorithms to skip
    pub(crate) skiplist: Vec<RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>>,
    /// Optional settings for linear regression
    pub(crate) linear_settings: Option<LinearRegressionParameters>,
    /// Optional settings for lasso regression
    pub(crate) lasso_settings: Option<LassoParameters>,
    /// Optional settings for ridge regression
    pub(crate) ridge_settings: Option<RidgeRegressionParameters<INPUT>>,
    /// Optional settings for elastic net
    pub(crate) elastic_net_settings: Option<ElasticNetParameters>,
    /// Optional settings for decision tree regressor
    pub(crate) decision_tree_regressor_settings: Option<DecisionTreeRegressorParameters>,
    /// Optional settings for random forest regressor
    pub(crate) random_forest_regressor_settings: Option<RandomForestRegressorParameters>,
    /// Optional settings for KNN regressor
    pub(crate) knn_regressor_settings: Option<KNNParameters>,
    /// Optional settings for support vector regressor
    pub(crate) svr_settings: Option<SVRParameters>,
    /// Optional settings for gradient boosting regressor
    pub(crate) xgboost_settings: Option<XGRegressorParameters>,
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + 'static,
    OUTPUT: FloatNumber + 'static,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>
        + 'static,
    OutputArray: Array1<OUTPUT> + 'static,
{
    fn default() -> Self {
        Self {
            supervised: SupervisedSettings {
                sort_by: Metric::RSquared,
                ..SupervisedSettings::default()
            },
            skiplist: vec![],
            linear_settings: Some(LinearRegressionParameters::default()),
            lasso_settings: Some(LassoParameters::default()),
            ridge_settings: Some(RidgeRegressionParameters::default()),
            elastic_net_settings: Some(ElasticNetParameters::default()),
            decision_tree_regressor_settings: Some(DecisionTreeRegressorParameters::default()),
            random_forest_regressor_settings: Some(RandomForestRegressorParameters::default()),
            knn_regressor_settings: Some(KNNParameters::default()),
            svr_settings: Some(SVRParameters::default()),
            xgboost_settings: Some(XGRegressorParameters::default()),
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + Number + 'static,
    OUTPUT: FloatNumber + Number + 'static,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>
        + 'static,
    OutputArray: Array1<OUTPUT> + 'static,
{
    /// Retrieve the metric function for regression tasks.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsError`] if no metric is set or if the metric is unsupported.
    pub fn get_metric(&self) -> Result<fn(&OutputArray, &OutputArray) -> f64, SettingsError> {
        match self.supervised.sort_by {
            Metric::RSquared => Ok(r2),
            Metric::MeanAbsoluteError => Ok(mean_absolute_error),
            Metric::MeanSquaredError => Ok(mean_squared_error),
            Metric::Accuracy => Err(SettingsError::UnsupportedMetric(Metric::Accuracy)),
            Metric::None => Err(SettingsError::MetricNotSet),
        }
    }

    /// Specify algorithms that shouldn't be included in comparison
    #[must_use]
    pub fn skip(
        mut self,
        skip: RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Specify only one algorithm to train
    #[must_use]
    pub fn only(
        mut self,
        only: &RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> Self {
        self.skiplist = RegressionAlgorithm::all_algorithms(&self)
            .into_iter()
            .filter(|algo| algo != only)
            .collect();
        self
    }

    with_settings_methods! {
        /// Specify settings for linear regression
        with_linear_settings, linear_settings, LinearRegressionParameters;
        /// Specify settings for lasso regression
        with_lasso_settings, lasso_settings, LassoParameters;
        /// Specify settings for ridge regression
        with_ridge_settings, ridge_settings, RidgeRegressionParameters<INPUT>;
        /// Specify settings for elastic net
        with_elastic_net_settings, elastic_net_settings, ElasticNetParameters;
        /// Specify settings for KNN regressor
        with_knn_regressor_settings, knn_regressor_settings, KNNParameters;
        /// Specify settings for random forest regressor
        with_random_forest_regressor_settings, random_forest_regressor_settings, RandomForestRegressorParameters;
        /// Specify settings for decision tree regressor
        with_decision_tree_regressor_settings, decision_tree_regressor_settings, DecisionTreeRegressorParameters;
        /// Specify settings for support vector regressor
        with_svr_settings, svr_settings, SVRParameters;
        /// Specify settings for gradient boosting regressor
        with_xgboost_settings, xgboost_settings, XGRegressorParameters;
    }

    /// Disable the support vector regressor by removing its settings.
    #[must_use]
    pub fn without_svr_settings(mut self) -> Self {
        self.svr_settings = None;
        self
    }

    /// Set the number of folds for cross-validation.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::RegressionSettings;
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    ///     .with_number_of_folds(5);
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
    /// use automl::settings::RegressionSettings;
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    ///     .shuffle_data(true);
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
    /// use automl::settings::RegressionSettings;
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    ///     .verbose(true);
    /// ```
    #[must_use]
    pub fn verbose(self, verbose: bool) -> Self {
        <Self as WithSupervisedSettings>::verbose(self, verbose)
    }

    /// Specify preprocessing strategy.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::{PreProcessing, RegressionSettings};
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
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
    /// use automl::settings::{FinalAlgorithm, RegressionSettings};
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
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
    /// use automl::settings::{Metric, RegressionSettings};
    /// use automl::DenseMatrix;
    /// let settings = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    ///     .sorted_by(Metric::RSquared);
    /// ```
    #[must_use]
    pub fn sorted_by(self, sort_by: Metric) -> Self {
        <Self as WithSupervisedSettings>::sorted_by(self, sort_by)
    }

    /// Create a [`KFold`] configuration from these settings.
    ///
    /// # Examples
    /// ```
    /// use automl::settings::RegressionSettings;
    /// use automl::DenseMatrix;
    /// let folds = RegressionSettings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default()
    ///     .get_kfolds();
    /// assert_eq!(folds.n_splits, 10);
    /// ```
    #[must_use]
    pub fn get_kfolds(&self) -> KFold {
        <Self as WithSupervisedSettings>::get_kfolds(self)
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> WithSupervisedSettings
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + 'static,
    OUTPUT: FloatNumber + 'static,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>
        + 'static,
    OutputArray: Array1<OUTPUT> + 'static,
{
    fn supervised(&self) -> &SupervisedSettings {
        &self.supervised
    }

    fn supervised_mut(&mut self) -> &mut SupervisedSettings {
        &mut self.supervised
    }
}

mod serde_impls {
    use super::RegressionSettings;
    use crate::algorithms::RegressionAlgorithm;
    use serde::de::{self, DeserializeOwned, MapAccess, Visitor};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use smartcore::linalg::basic::arrays::Array1;
    use smartcore::linalg::traits::{
        cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable,
        svd::SVDDecomposable,
    };
    use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};
    use std::fmt;
    use std::marker::PhantomData;

    use super::{
        DecisionTreeRegressorParameters, ElasticNetParameters, KNNParameters, LassoParameters,
        LinearRegressionParameters, RandomForestRegressorParameters, RidgeRegressionParameters,
        SVRParameters, SupervisedSettings, XGRegressorParameters,
    };
    use crate::settings::Objective;

    const FIELD_COUNT: usize = 11;

    fn algorithm_to_name<INPUT, OUTPUT, InputArray, OutputArray>(
        algorithm: &RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>,
    ) -> &'static str
    where
        INPUT: RealNumber + FloatNumber + 'static,
        OUTPUT: FloatNumber + 'static,
        InputArray: smartcore::linalg::basic::arrays::Array2<INPUT>
            + QRDecomposable<INPUT>
            + SVDDecomposable<INPUT>
            + EVDDecomposable<INPUT>
            + CholeskyDecomposable<INPUT>
            + Clone
            + Sized
            + 'static,
        OutputArray: Array1<OUTPUT> + Clone + Sized + 'static,
    {
        match algorithm {
            RegressionAlgorithm::DecisionTreeRegressor(_) => "decision_tree_regressor",
            RegressionAlgorithm::RandomForestRegressor(_) => "random_forest_regressor",
            RegressionAlgorithm::Linear(_) => "linear_regressor",
            RegressionAlgorithm::Ridge(_) => "ridge_regressor",
            RegressionAlgorithm::Lasso(_) => "lasso_regressor",
            RegressionAlgorithm::ElasticNet(_) => "elastic_net_regressor",
            RegressionAlgorithm::KNNRegressor(_) => "knn_regressor",
            RegressionAlgorithm::SupportVectorRegressor(_) => "support_vector_regressor",
            RegressionAlgorithm::XGBoostRegressor(_) => "xgboost_regressor",
        }
    }

    fn algorithm_from_name<INPUT, OUTPUT, InputArray, OutputArray>(
        name: &str,
    ) -> Result<RegressionAlgorithm<INPUT, OUTPUT, InputArray, OutputArray>, String>
    where
        INPUT: RealNumber + FloatNumber + 'static,
        OUTPUT: FloatNumber + 'static,
        InputArray: smartcore::linalg::basic::arrays::Array2<INPUT>
            + QRDecomposable<INPUT>
            + SVDDecomposable<INPUT>
            + EVDDecomposable<INPUT>
            + CholeskyDecomposable<INPUT>
            + Clone
            + Sized
            + 'static,
        OutputArray: Array1<OUTPUT> + Clone + Sized + 'static,
    {
        match name {
            "decision_tree_regressor" => Ok(RegressionAlgorithm::default_decision_tree()),
            "random_forest_regressor" => Ok(RegressionAlgorithm::default_random_forest()),
            "linear_regressor" => Ok(RegressionAlgorithm::default_linear()),
            "ridge_regressor" => Ok(RegressionAlgorithm::default_ridge()),
            "lasso_regressor" => Ok(RegressionAlgorithm::default_lasso()),
            "elastic_net_regressor" => Ok(RegressionAlgorithm::default_elastic_net()),
            "knn_regressor" => Ok(RegressionAlgorithm::default_knn_regressor()),
            "support_vector_regressor" => {
                Ok(RegressionAlgorithm::default_support_vector_regressor())
            }
            "xgboost_regressor" => Ok(RegressionAlgorithm::default_xgboost_regressor()),
            other => Err(format!("unknown regression algorithm '{other}'")),
        }
    }

    fn objective_to_str(objective: &Objective) -> &'static str {
        match objective {
            Objective::MeanSquaredError => "MeanSquaredError",
        }
    }

    fn objective_from_str(value: &str) -> Result<Objective, String> {
        match value {
            "MeanSquaredError" => Ok(Objective::MeanSquaredError),
            other => Err(format!("unknown xgboost objective '{other}'")),
        }
    }

    struct SerializableXGBoostParameters<'a>(&'a XGRegressorParameters);

    impl Serialize for SerializableXGBoostParameters<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let params = self.0;
            let mut state = serializer.serialize_struct("XGRegressorParameters", 10)?;
            state.serialize_field("n_estimators", &params.n_estimators)?;
            state.serialize_field("max_depth", &params.max_depth)?;
            state.serialize_field("learning_rate", &params.learning_rate)?;
            state.serialize_field("min_child_weight", &params.min_child_weight)?;
            state.serialize_field("lambda", &params.lambda)?;
            state.serialize_field("gamma", &params.gamma)?;
            state.serialize_field("base_score", &params.base_score)?;
            state.serialize_field("subsample", &params.subsample)?;
            state.serialize_field("seed", &params.seed)?;
            state.serialize_field("objective", &objective_to_str(&params.objective))?;
            state.end()
        }
    }

    struct DeserializableXGBoostParameters(pub XGRegressorParameters);

    impl<'de> Deserialize<'de> for DeserializableXGBoostParameters {
        #[allow(clippy::too_many_lines)]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                NEstimators,
                MaxDepth,
                LearningRate,
                MinChildWeight,
                Lambda,
                Gamma,
                BaseScore,
                Subsample,
                Seed,
                Objective,
            }

            impl<'de> Deserialize<'de> for Field {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    struct FieldVisitor;

                    impl Visitor<'_> for FieldVisitor {
                        type Value = Field;

                        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                            formatter.write_str("a field in XGRegressorParameters")
                        }

                        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where
                            E: de::Error,
                        {
                            match value {
                                "n_estimators" => Ok(Field::NEstimators),
                                "max_depth" => Ok(Field::MaxDepth),
                                "learning_rate" => Ok(Field::LearningRate),
                                "min_child_weight" => Ok(Field::MinChildWeight),
                                "lambda" => Ok(Field::Lambda),
                                "gamma" => Ok(Field::Gamma),
                                "base_score" => Ok(Field::BaseScore),
                                "subsample" => Ok(Field::Subsample),
                                "seed" => Ok(Field::Seed),
                                "objective" => Ok(Field::Objective),
                                other => Err(de::Error::unknown_field(other, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct ParamsVisitor;

            impl<'de> Visitor<'de> for ParamsVisitor {
                type Value = DeserializableXGBoostParameters;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a map describing XGRegressorParameters")
                }

                #[allow(clippy::too_many_lines)]
                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: MapAccess<'de>,
                {
                    let mut n_estimators = None;
                    let mut max_depth = None;
                    let mut learning_rate = None;
                    let mut min_child_weight = None;
                    let mut lambda = None;
                    let mut gamma = None;
                    let mut base_score = None;
                    let mut subsample = None;
                    let mut seed = None;
                    let mut objective: Option<Objective> = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::NEstimators => {
                                if n_estimators.is_some() {
                                    return Err(de::Error::duplicate_field("n_estimators"));
                                }
                                n_estimators = Some(map.next_value()?);
                            }
                            Field::MaxDepth => {
                                if max_depth.is_some() {
                                    return Err(de::Error::duplicate_field("max_depth"));
                                }
                                max_depth = Some(map.next_value()?);
                            }
                            Field::LearningRate => {
                                if learning_rate.is_some() {
                                    return Err(de::Error::duplicate_field("learning_rate"));
                                }
                                learning_rate = Some(map.next_value()?);
                            }
                            Field::MinChildWeight => {
                                if min_child_weight.is_some() {
                                    return Err(de::Error::duplicate_field("min_child_weight"));
                                }
                                min_child_weight = Some(map.next_value()?);
                            }
                            Field::Lambda => {
                                if lambda.is_some() {
                                    return Err(de::Error::duplicate_field("lambda"));
                                }
                                lambda = Some(map.next_value()?);
                            }
                            Field::Gamma => {
                                if gamma.is_some() {
                                    return Err(de::Error::duplicate_field("gamma"));
                                }
                                gamma = Some(map.next_value()?);
                            }
                            Field::BaseScore => {
                                if base_score.is_some() {
                                    return Err(de::Error::duplicate_field("base_score"));
                                }
                                base_score = Some(map.next_value()?);
                            }
                            Field::Subsample => {
                                if subsample.is_some() {
                                    return Err(de::Error::duplicate_field("subsample"));
                                }
                                subsample = Some(map.next_value()?);
                            }
                            Field::Seed => {
                                if seed.is_some() {
                                    return Err(de::Error::duplicate_field("seed"));
                                }
                                seed = Some(map.next_value()?);
                            }
                            Field::Objective => {
                                if objective.is_some() {
                                    return Err(de::Error::duplicate_field("objective"));
                                }
                                let value: String = map.next_value()?;
                                let parsed =
                                    objective_from_str(&value).map_err(de::Error::custom)?;
                                objective = Some(parsed);
                            }
                        }
                    }

                    let mut params = XGRegressorParameters::default();
                    if let Some(value) = n_estimators {
                        params.n_estimators = value;
                    }
                    if let Some(value) = max_depth {
                        params.max_depth = value;
                    }
                    if let Some(value) = learning_rate {
                        params.learning_rate = value;
                    }
                    if let Some(value) = min_child_weight {
                        params.min_child_weight = value;
                    }
                    if let Some(value) = lambda {
                        params.lambda = value;
                    }
                    if let Some(value) = gamma {
                        params.gamma = value;
                    }
                    if let Some(value) = base_score {
                        params.base_score = value;
                    }
                    if let Some(value) = subsample {
                        params.subsample = value;
                    }
                    if let Some(value) = seed {
                        params.seed = value;
                    }
                    if let Some(value) = objective {
                        params.objective = value;
                    }
                    Ok(DeserializableXGBoostParameters(params))
                }
            }

            const FIELDS: &[&str] = &[
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_child_weight",
                "lambda",
                "gamma",
                "base_score",
                "subsample",
                "seed",
                "objective",
            ];

            deserializer.deserialize_struct("XGRegressorParameters", FIELDS, ParamsVisitor)
        }
    }

    impl<INPUT, OUTPUT, InputArray, OutputArray> Serialize
        for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
    where
        INPUT: FloatNumber + RealNumber + Serialize + 'static,
        OUTPUT: FloatNumber + 'static,
        InputArray: smartcore::linalg::basic::arrays::Array2<INPUT>
            + QRDecomposable<INPUT>
            + SVDDecomposable<INPUT>
            + EVDDecomposable<INPUT>
            + CholeskyDecomposable<INPUT>
            + Clone
            + Sized
            + 'static,
        OutputArray: Array1<OUTPUT> + Clone + Sized + 'static,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut state = serializer.serialize_struct("RegressionSettings", FIELD_COUNT)?;
            state.serialize_field("supervised", &self.supervised)?;
            let skiplist: Vec<String> = self
                .skiplist
                .iter()
                .map(|algo| algorithm_to_name(algo).to_string())
                .collect();
            state.serialize_field("skiplist", &skiplist)?;
            state.serialize_field("linear_settings", &self.linear_settings)?;
            state.serialize_field("lasso_settings", &self.lasso_settings)?;
            state.serialize_field("ridge_settings", &self.ridge_settings)?;
            state.serialize_field("elastic_net_settings", &self.elastic_net_settings)?;
            state.serialize_field(
                "decision_tree_regressor_settings",
                &self.decision_tree_regressor_settings,
            )?;
            state.serialize_field(
                "random_forest_regressor_settings",
                &self.random_forest_regressor_settings,
            )?;
            state.serialize_field("knn_regressor_settings", &self.knn_regressor_settings)?;
            state.serialize_field("svr_settings", &self.svr_settings)?;
            match &self.xgboost_settings {
                Some(params) => state.serialize_field(
                    "xgboost_settings",
                    &Some(SerializableXGBoostParameters(params)),
                )?,
                None => state.serialize_field::<Option<SerializableXGBoostParameters>>(
                    "xgboost_settings",
                    &None,
                )?,
            }
            state.end()
        }
    }

    impl<'de, INPUT, OUTPUT, InputArray, OutputArray> Deserialize<'de>
        for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
    where
        INPUT: FloatNumber + RealNumber + Number + DeserializeOwned + 'static,
        OUTPUT: FloatNumber + Number + 'static,
        InputArray: smartcore::linalg::basic::arrays::Array2<INPUT>
            + QRDecomposable<INPUT>
            + SVDDecomposable<INPUT>
            + EVDDecomposable<INPUT>
            + CholeskyDecomposable<INPUT>
            + Clone
            + Sized
            + 'static,
        OutputArray: Array1<OUTPUT> + Clone + Sized + 'static,
    {
        #[allow(clippy::too_many_lines)]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                Supervised,
                Skiplist,
                LinearSettings,
                LassoSettings,
                RidgeSettings,
                ElasticNetSettings,
                DecisionTreeRegressorSettings,
                RandomForestRegressorSettings,
                KnnRegressorSettings,
                SvrSettings,
                XgboostSettings,
            }

            impl<'de> Deserialize<'de> for Field {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    struct FieldVisitor;

                    impl Visitor<'_> for FieldVisitor {
                        type Value = Field;

                        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                            formatter.write_str("a field in RegressionSettings")
                        }

                        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where
                            E: de::Error,
                        {
                            match value {
                                "supervised" => Ok(Field::Supervised),
                                "skiplist" => Ok(Field::Skiplist),
                                "linear_settings" => Ok(Field::LinearSettings),
                                "lasso_settings" => Ok(Field::LassoSettings),
                                "ridge_settings" => Ok(Field::RidgeSettings),
                                "elastic_net_settings" => Ok(Field::ElasticNetSettings),
                                "decision_tree_regressor_settings" => {
                                    Ok(Field::DecisionTreeRegressorSettings)
                                }
                                "random_forest_regressor_settings" => {
                                    Ok(Field::RandomForestRegressorSettings)
                                }
                                "knn_regressor_settings" => Ok(Field::KnnRegressorSettings),
                                "svr_settings" => Ok(Field::SvrSettings),
                                "xgboost_settings" => Ok(Field::XgboostSettings),
                                other => Err(de::Error::unknown_field(other, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct RegressionSettingsVisitor<INPUT, OUTPUT, InputArray, OutputArray> {
                _marker: PhantomData<(INPUT, OUTPUT, InputArray, OutputArray)>,
            }

            impl<'de, INPUT, OUTPUT, InputArray, OutputArray> Visitor<'de>
                for RegressionSettingsVisitor<INPUT, OUTPUT, InputArray, OutputArray>
            where
                INPUT: FloatNumber + RealNumber + Number + DeserializeOwned + 'static,
                OUTPUT: FloatNumber + Number + 'static,
                InputArray: smartcore::linalg::basic::arrays::Array2<INPUT>
                    + QRDecomposable<INPUT>
                    + SVDDecomposable<INPUT>
                    + EVDDecomposable<INPUT>
                    + CholeskyDecomposable<INPUT>
                    + Clone
                    + Sized
                    + 'static,
                OutputArray: Array1<OUTPUT> + Clone + Sized + 'static,
            {
                type Value = RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a map describing RegressionSettings")
                }

                #[allow(clippy::too_many_lines)]
                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: MapAccess<'de>,
                {
                    let mut supervised: Option<SupervisedSettings> = None;
                    let mut skiplist: Option<Vec<String>> = None;
                    let mut linear_settings: Option<Option<LinearRegressionParameters>> = None;
                    let mut lasso_settings: Option<Option<LassoParameters>> = None;
                    let mut ridge_settings: Option<Option<RidgeRegressionParameters<INPUT>>> = None;
                    let mut elastic_net_settings: Option<Option<ElasticNetParameters>> = None;
                    let mut decision_tree_regressor_settings: Option<
                        Option<DecisionTreeRegressorParameters>,
                    > = None;
                    let mut random_forest_regressor_settings: Option<
                        Option<RandomForestRegressorParameters>,
                    > = None;
                    let mut knn_regressor_settings: Option<Option<KNNParameters>> = None;
                    let mut svr_settings: Option<Option<SVRParameters>> = None;
                    let mut xgboost_settings: Option<Option<XGRegressorParameters>> = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::Supervised => {
                                if supervised.is_some() {
                                    return Err(de::Error::duplicate_field("supervised"));
                                }
                                supervised = Some(map.next_value()?);
                            }
                            Field::Skiplist => {
                                if skiplist.is_some() {
                                    return Err(de::Error::duplicate_field("skiplist"));
                                }
                                skiplist = Some(map.next_value()?);
                            }
                            Field::LinearSettings => {
                                if linear_settings.is_some() {
                                    return Err(de::Error::duplicate_field("linear_settings"));
                                }
                                linear_settings = Some(map.next_value()?);
                            }
                            Field::LassoSettings => {
                                if lasso_settings.is_some() {
                                    return Err(de::Error::duplicate_field("lasso_settings"));
                                }
                                lasso_settings = Some(map.next_value()?);
                            }
                            Field::RidgeSettings => {
                                if ridge_settings.is_some() {
                                    return Err(de::Error::duplicate_field("ridge_settings"));
                                }
                                ridge_settings = Some(map.next_value()?);
                            }
                            Field::ElasticNetSettings => {
                                if elastic_net_settings.is_some() {
                                    return Err(de::Error::duplicate_field("elastic_net_settings"));
                                }
                                elastic_net_settings = Some(map.next_value()?);
                            }
                            Field::DecisionTreeRegressorSettings => {
                                if decision_tree_regressor_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "decision_tree_regressor_settings",
                                    ));
                                }
                                decision_tree_regressor_settings = Some(map.next_value()?);
                            }
                            Field::RandomForestRegressorSettings => {
                                if random_forest_regressor_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "random_forest_regressor_settings",
                                    ));
                                }
                                random_forest_regressor_settings = Some(map.next_value()?);
                            }
                            Field::KnnRegressorSettings => {
                                if knn_regressor_settings.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "knn_regressor_settings",
                                    ));
                                }
                                knn_regressor_settings = Some(map.next_value()?);
                            }
                            Field::SvrSettings => {
                                if svr_settings.is_some() {
                                    return Err(de::Error::duplicate_field("svr_settings"));
                                }
                                svr_settings = Some(map.next_value()?);
                            }
                            Field::XgboostSettings => {
                                if xgboost_settings.is_some() {
                                    return Err(de::Error::duplicate_field("xgboost_settings"));
                                }
                                let value: Option<DeserializableXGBoostParameters> =
                                    map.next_value()?;
                                xgboost_settings = Some(value.map(|wrapper| wrapper.0));
                            }
                        }
                    }

                    let mut settings = RegressionSettings::default();
                    if let Some(value) = supervised {
                        settings.supervised = value;
                    }
                    if let Some(names) = skiplist {
                        let mut converted = Vec::with_capacity(names.len());
                        for name in names {
                            let algorithm =
                                algorithm_from_name::<INPUT, OUTPUT, InputArray, OutputArray>(
                                    &name,
                                )
                                .map_err(de::Error::custom)?;
                            converted.push(algorithm);
                        }
                        settings.skiplist = converted;
                    }
                    if let Some(value) = linear_settings {
                        settings.linear_settings = value;
                    }
                    if let Some(value) = lasso_settings {
                        settings.lasso_settings = value;
                    }
                    if let Some(value) = ridge_settings {
                        settings.ridge_settings = value;
                    }
                    if let Some(value) = elastic_net_settings {
                        settings.elastic_net_settings = value;
                    }
                    if let Some(value) = decision_tree_regressor_settings {
                        settings.decision_tree_regressor_settings = value;
                    }
                    if let Some(value) = random_forest_regressor_settings {
                        settings.random_forest_regressor_settings = value;
                    }
                    if let Some(value) = knn_regressor_settings {
                        settings.knn_regressor_settings = value;
                    }
                    if let Some(value) = svr_settings {
                        settings.svr_settings = value;
                    }
                    if let Some(value) = xgboost_settings {
                        settings.xgboost_settings = value;
                    }
                    Ok(settings)
                }
            }

            const FIELDS: &[&str] = &[
                "supervised",
                "skiplist",
                "linear_settings",
                "lasso_settings",
                "ridge_settings",
                "elastic_net_settings",
                "decision_tree_regressor_settings",
                "random_forest_regressor_settings",
                "knn_regressor_settings",
                "svr_settings",
                "xgboost_settings",
            ];

            deserializer.deserialize_struct(
                "RegressionSettings",
                FIELDS,
                RegressionSettingsVisitor {
                    _marker: PhantomData,
                },
            )
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + 'static,
    OUTPUT: FloatNumber + 'static,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>
        + 'static,
    OutputArray: Array1<OUTPUT> + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Regression settings: sorted by {}",
            self.supervised.sort_by
        )
    }
}
