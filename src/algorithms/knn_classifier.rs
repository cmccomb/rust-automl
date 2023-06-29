//! KNN Classifier

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    math::distance::{
        euclidian::Euclidian, hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan,
        minkowski::Minkowski, Distances,
    },
    model_selection::{cross_validate, CrossValidationResult},
    neighbors::knn_classifier::{
        KNNClassifier, KNNClassifierParameters as SmartcoreKNNClassifierParameters,
    },
};

use crate::{Algorithm, Distance, Settings};

/// The KNN Classifier.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/neighbors.html#classification)
/// for a more in-depth description of the algorithm.
pub(crate) struct KNNClassifierWrapper {}

impl super::ModelWrapper for KNNClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        let cv = match settings.knn_classifier_settings.as_ref().unwrap().distance {
            Distance::Euclidean => cross_validate(
                KNNClassifier::fit,
                x,
                y,
                SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::euclidian()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Manhattan => cross_validate(
                KNNClassifier::fit,
                x,
                y,
                SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::manhattan()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Minkowski(p) => cross_validate(
                KNNClassifier::fit,
                x,
                y,
                SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::minkowski(p)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Mahalanobis => cross_validate(
                KNNClassifier::fit,
                x,
                y,
                SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::mahalanobis(x)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Hamming => cross_validate(
                KNNClassifier::fit,
                x,
                y,
                SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::hamming()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
        };

        (cv, Algorithm::KNNClassifier)
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        match settings.knn_classifier_settings.as_ref().unwrap().distance {
            Distance::Euclidean => {
                let params = SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::euclidian());
                bincode::serialize(&KNNClassifier::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Manhattan => {
                let params = SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::manhattan());
                bincode::serialize(&KNNClassifier::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Minkowski(p) => {
                let params = SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::minkowski(p));
                bincode::serialize(&KNNClassifier::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Mahalanobis => {
                let params = SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::mahalanobis(x));
                bincode::serialize(&KNNClassifier::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Hamming => {
                let params = SmartcoreKNNClassifierParameters::default()
                    .with_k(settings.knn_classifier_settings.as_ref().unwrap().k)
                    .with_weight(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_algorithm(
                        settings
                            .knn_classifier_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_distance(Distances::hamming());
                bincode::serialize(&KNNClassifier::fit(x, y, params).unwrap()).unwrap()
            }
        }
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        match settings.knn_classifier_settings.as_ref().unwrap().distance {
            Distance::Euclidean => {
                let model: KNNClassifier<f32, Euclidian> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Manhattan => {
                let model: KNNClassifier<f32, Manhattan> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Minkowski(_) => {
                let model: KNNClassifier<f32, Minkowski> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Mahalanobis => {
                let model: KNNClassifier<f32, Mahalanobis<f32, DenseMatrix<f32>>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Hamming => {
                let model: KNNClassifier<f32, Hamming> = bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}
