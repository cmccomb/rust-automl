use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::cross_validate;
use smartcore::svm::svc::SVC;

use crate::{Algorithm, Kernel, Settings, SmartcoreSVCParameters};
use smartcore::model_selection::CrossValidationResult;
use smartcore::svm::Kernels;

pub(crate) struct SupportVectorClassifierWrapper {}

impl super::ModelWrapper for SupportVectorClassifierWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        let cv = match settings.svc_settings.as_ref().unwrap().kernel {
            Kernel::Linear => cross_validate(
                SVC::fit,
                x,
                y,
                SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::linear()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::Polynomial(degree, gamma, coef) => cross_validate(
                SVC::fit,
                x,
                y,
                SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::polynomial(degree, gamma, coef)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::RBF(gamma) => cross_validate(
                SVC::fit,
                x,
                y,
                SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::rbf(gamma)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::Sigmoid(gamma, coef) => cross_validate(
                SVC::fit,
                x,
                y,
                SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::sigmoid(gamma, coef)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
        };
        (cv, Algorithm::SVC)
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        todo!()
    }
}
