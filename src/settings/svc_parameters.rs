pub use crate::utils::Kernel;

/// Parameters for support vector classification
pub struct SVCParameters {
    pub(crate) epoch: usize,
    pub(crate) c: f32,
    pub(crate) tol: f32,
    pub(crate) kernel: Kernel,
}

impl SVCParameters {
    /// Define the value of epsilon to use in the epsilon-SVR model.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Define the regulation penalty to use with the SVR Model
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Define the convergence tolerance to use with the SVR model
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Define which kernel to use with the SVR model
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
