use std::fmt::{Display, Formatter};

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
