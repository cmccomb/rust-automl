//! Kernel options for support vector machines.

use std::fmt::{Display, Formatter};

use smartcore::error::{Failed, FailedError};
use smartcore::svm::Kernels as SmartcoreKernels;

/// Smartcore kernel conversion result containing both the enum representation and a boxed kernel
/// function suitable for Smartcore APIs.
pub struct SmartcoreKernel {
    /// Smartcore kernel enum configured with validated parameters.
    pub kernel: SmartcoreKernels,
    /// Boxed kernel function implementing [`smartcore::svm::Kernel`].
    pub function: Box<dyn smartcore::svm::Kernel>,
}

/// Kernel options for use with support vector machines
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Kernel {
    /// Linear Kernel
    Linear,

    /// Polynomial kernel
    Polynomial(f64, f64, f64),

    /// Radial basis function kernel
    RBF(f64),

    /// Sigmoid kernel
    Sigmoid(f64, f64),
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Polynomial(degree, gamma, coef) => write!(
                f,
                "Polynomial\n    degree = {degree}\n    gamma = {gamma}\n    coef = {coef}"
            ),
            Self::RBF(gamma) => write!(f, "RBF\n    gamma = {gamma}"),
            Self::Sigmoid(gamma, coef) => {
                write!(f, "Sigmoid\n    gamma = {gamma}\n    coef = {coef}")
            }
        }
    }
}

impl Kernel {
    /// Convert the enum variant into Smartcore kernel components.
    ///
    /// # Errors
    ///
    /// Returns [`Failed`] when kernel parameters are invalid (for example, NaN or non-positive
    /// values where Smartcore requires strictly positive parameters).
    pub fn to_smartcore(&self) -> Result<SmartcoreKernel, Failed> {
        match self {
            Self::Linear => {
                let kernel = SmartcoreKernels::linear();
                Ok(SmartcoreKernel {
                    function: Box::new(kernel.clone()),
                    kernel,
                })
            }
            Self::Polynomial(degree, gamma, coef0) => {
                validate_positive(*degree, "polynomial degree")?;
                validate_positive(*gamma, "polynomial gamma")?;
                validate_finite(*coef0, "polynomial coef0")?;
                let kernel = SmartcoreKernels::polynomial()
                    .with_degree(*degree)
                    .with_gamma(*gamma)
                    .with_coef0(*coef0);
                Ok(SmartcoreKernel {
                    function: Box::new(kernel.clone()),
                    kernel,
                })
            }
            Self::RBF(gamma) => {
                validate_positive(*gamma, "RBF gamma")?;
                let kernel = SmartcoreKernels::rbf().with_gamma(*gamma);
                Ok(SmartcoreKernel {
                    function: Box::new(kernel.clone()),
                    kernel,
                })
            }
            Self::Sigmoid(gamma, coef0) => {
                validate_finite(*gamma, "sigmoid gamma")?;
                validate_finite(*coef0, "sigmoid coef0")?;
                let kernel = SmartcoreKernels::sigmoid()
                    .with_gamma(*gamma)
                    .with_coef0(*coef0);
                Ok(SmartcoreKernel {
                    function: Box::new(kernel.clone()),
                    kernel,
                })
            }
        }
    }
}

fn validate_finite(value: f64, name: &str) -> Result<(), Failed> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Failed::because(
            FailedError::ParametersError,
            &format!("{name} must be finite"),
        ))
    }
}

fn validate_positive(value: f64, name: &str) -> Result<(), Failed> {
    validate_finite(value, name)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(Failed::because(
            FailedError::ParametersError,
            &format!("{name} must be positive"),
        ))
    }
}
