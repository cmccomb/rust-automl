//! Settings for regression models

#![allow(clippy::struct_field_names)]

use super::{
    DecisionTreeRegressorParameters, ElasticNetParameters, KNNParameters, LassoParameters,
    LinearRegressionParameters, Metric, RandomForestRegressorParameters, RidgeRegressionParameters,
    SupervisedSettings, WithSupervisedSettings,
};
use crate::algorithms::RegressionAlgorithm;
use crate::settings::macros::with_settings_methods;

use smartcore::linalg::basic::arrays::Array1;
use smartcore::linalg::traits::{
    cholesky::CholeskyDecomposable, evd::EVDDecomposable, qr::QRDecomposable, svd::SVDDecomposable,
};
use smartcore::metrics::{mean_absolute_error, mean_squared_error, r2};
use smartcore::numbers::{basenum::Number, floatnum::FloatNumber, realnum::RealNumber};
use std::fmt::{Display, Formatter};

/// Settings for regression models.
///
/// Any algorithms in the `skiplist` member will be skipped during training.
pub struct RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
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
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Default
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
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
        }
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray>
    RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber + Number,
    OUTPUT: FloatNumber + Number,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    /// Get the metric function
    pub(crate) fn get_metric(&self) -> fn(&OutputArray, &OutputArray) -> f64 {
        match self.supervised.sort_by {
            Metric::RSquared => r2,
            Metric::MeanAbsoluteError => mean_absolute_error,
            Metric::MeanSquaredError => mean_squared_error,
            Metric::Accuracy => panic!("Accuracy metric not supported for regression"),
            Metric::None => panic!("A metric must be set."),
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
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> WithSupervisedSettings
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    fn supervised(&self) -> &SupervisedSettings {
        &self.supervised
    }

    fn supervised_mut(&mut self) -> &mut SupervisedSettings {
        &mut self.supervised
    }
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for RegressionSettings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT>
        + SVDDecomposable<INPUT>
        + EVDDecomposable<INPUT>
        + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Regression settings: sorted by {}",
            self.supervised.sort_by
        )
    }
}
