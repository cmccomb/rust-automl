//! Helpers for constructing Smartcore KNN parameter structs from shared settings.

use super::{KNNClassifierParameters, KNNRegressorParameters};
use smartcore::metrics::distance::Distance as SmartDistance;
use smartcore::neighbors::knn_classifier::KNNClassifierParameters as SmartKNNClassifierParameters;
use smartcore::neighbors::knn_regressor::KNNRegressorParameters as SmartKNNRegressorParameters;
use smartcore::numbers::basenum::Number;

/// Build Smartcore KNN regressor parameters from common settings.
///
/// # Examples
/// ```
/// use automl::settings::{build_knn_regressor_parameters, KNNRegressorParameters};
/// use smartcore::metrics::distance::euclidian::Euclidian;
///
/// let settings = KNNRegressorParameters::default().with_k(5);
/// let params = build_knn_regressor_parameters::<f64, _>(&settings, Euclidian::new());
/// assert_eq!(params.k, 5);
/// ```
pub fn build_knn_regressor_parameters<T, D>(
    settings: &KNNRegressorParameters,
    distance: D,
) -> SmartKNNRegressorParameters<T, D>
where
    T: Number,
    D: SmartDistance<Vec<T>>,
{
    SmartKNNRegressorParameters::<T, _>::default()
        .with_k(settings.k)
        .with_algorithm(settings.algorithm.clone())
        .with_weight(settings.weight.clone())
        .with_distance(distance)
}

/// Build Smartcore KNN classifier parameters from common settings.
///
/// # Examples
/// ```
/// use automl::settings::{build_knn_classifier_parameters, KNNClassifierParameters};
/// use smartcore::metrics::distance::euclidian::Euclidian;
///
/// let settings = KNNClassifierParameters::default().with_k(7);
/// let params = build_knn_classifier_parameters::<f64, _>(&settings, Euclidian::new());
/// assert_eq!(params.k, 7);
/// ```
pub fn build_knn_classifier_parameters<T, D>(
    settings: &KNNClassifierParameters,
    distance: D,
) -> SmartKNNClassifierParameters<T, D>
where
    T: Number,
    D: SmartDistance<Vec<T>>,
{
    SmartKNNClassifierParameters::<T, _>::default()
        .with_k(settings.k)
        .with_algorithm(settings.algorithm.clone())
        .with_weight(settings.weight.clone())
        .with_distance(distance)
}
