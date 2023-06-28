//! Support Vector Classification parameters

pub use crate::utils::Kernel;

/// Parameters for support vector classification
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SVCParameters {
    /// Number of epochs to use in the epsilon-SVC model
    pub(crate) epoch: usize,
    /// Regulation penalty to use with the SVC model
    pub(crate) c: f32,
    /// Convergence tolerance to use with the SVC model
    pub(crate) tol: f32,
    /// Kernel to use with the SVC model
    pub(crate) kernel: Kernel,
}

impl SVCParameters {
    /// Define the number of epochs to use in the epsilon-SVC model.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Define the regulation penalty to use with the SVC Model
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Define the convergence tolerance to use with the SVC model
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Define which kernel to use with the SVC model
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
}

impl Default for SVCParameters {
    fn default() -> Self {
        Self {
            epoch: 2,
            c: 1.0,
            tol: 1e-3,
            kernel: Kernel::Linear,
        }
    }
}
