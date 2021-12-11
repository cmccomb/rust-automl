use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    math::distance::{
        euclidian::Euclidian, hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan,
        minkowski::Minkowski, Distances,
    },
    model_selection::cross_validate,
    model_selection::CrossValidationResult,
    neighbors::knn_regressor::{
        KNNRegressor, KNNRegressorParameters as SmartcoreKNNRegressorParameters,
    },
};

use crate::{Algorithm, Distance, Settings};

pub(crate) struct KNNRegressorWrapper {}

impl super::ModelWrapper for KNNRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        let cv = match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::euclidian()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Manhattan => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::manhattan()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Minkowski(p) => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::minkowski(p)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Mahalanobis => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::mahalanobis(x)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Hamming => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::hamming()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
        };

        (cv, Algorithm::KNNRegressor)
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => {
                let params = SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::euclidian());

                bincode::serialize(&KNNRegressor::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Manhattan => {
                let params = SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::manhattan());

                bincode::serialize(&KNNRegressor::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Minkowski(p) => {
                let params = SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::minkowski(p));

                bincode::serialize(&KNNRegressor::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Mahalanobis => {
                let params = SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::mahalanobis(x));
                bincode::serialize(&KNNRegressor::fit(x, y, params).unwrap()).unwrap()
            }
            Distance::Hamming => {
                let params = SmartcoreKNNRegressorParameters::default()
                    .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
                    .with_algorithm(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .unwrap()
                            .algorithm
                            .clone(),
                    )
                    .with_weight(
                        settings
                            .knn_regressor_settings
                            .as_ref()
                            .as_ref()
                            .unwrap()
                            .weight
                            .clone(),
                    )
                    .with_distance(Distances::hamming());

                bincode::serialize(&KNNRegressor::fit(x, y, params).unwrap()).unwrap()
            }
        }
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => {
                let model: KNNRegressor<f32, Euclidian> =
                    bincode::deserialize(&*final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Manhattan => {
                let model: KNNRegressor<f32, Manhattan> =
                    bincode::deserialize(&*final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Minkowski(_) => {
                let model: KNNRegressor<f32, Minkowski> =
                    bincode::deserialize(&*final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Mahalanobis => {
                let model: KNNRegressor<f32, Mahalanobis<f32, DenseMatrix<f32>>> =
                    bincode::deserialize(&*final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Hamming => {
                let model: KNNRegressor<f32, Hamming> =
                    bincode::deserialize(&*final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}
