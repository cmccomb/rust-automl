//! Support Vector Classifier

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    model_selection::cross_validate,
    model_selection::CrossValidationResult,
    svm::{
        svc::{SVCParameters as SmartcoreSVCParameters, SVC},
        Kernels, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel,
    },
};

use crate::{Algorithm, Kernel, Settings};

/// The Support Vector Classifier.
/// 
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/svm.html#svm-classification)
/// for a more in-depth description of the algorithm.
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

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        match settings.svc_settings.as_ref().unwrap().kernel {
            Kernel::Linear => {
                let params = SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::linear());

                bincode::serialize(&SVC::fit(x, y, params).unwrap()).unwrap()
            }
            Kernel::Polynomial(degree, gamma, coef) => {
                let params = SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::polynomial(degree, gamma, coef));

                bincode::serialize(&SVC::fit(x, y, params).unwrap()).unwrap()
            }
            Kernel::RBF(gamma) => {
                let params = SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::rbf(gamma));

                bincode::serialize(&SVC::fit(x, y, params).unwrap()).unwrap()
            }
            Kernel::Sigmoid(gamma, coef) => {
                let params = SmartcoreSVCParameters::default()
                    .with_tol(settings.svc_settings.as_ref().unwrap().tol)
                    .with_c(settings.svc_settings.as_ref().unwrap().c)
                    .with_epoch(settings.svc_settings.as_ref().unwrap().epoch)
                    .with_kernel(Kernels::sigmoid(gamma, coef));

                bincode::serialize(&SVC::fit(x, y, params).unwrap()).unwrap()
            }
        }
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        match settings.svc_settings.as_ref().unwrap().kernel {
            Kernel::Linear => {
                let model: SVC<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::Polynomial(_, _, _) => {
                let model: SVC<f32, DenseMatrix<f32>, PolynomialKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::RBF(_) => {
                let model: SVC<f32, DenseMatrix<f32>, RBFKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::Sigmoid(_, _) => {
                let model: SVC<f32, DenseMatrix<f32>, SigmoidKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}
