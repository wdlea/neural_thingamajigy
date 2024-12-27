#[cfg(feature = "train")]
use nalgebra::{SMatrix, SVector};

/// Contains an activaton function and it's gradient
pub struct Activator {
    /// The activation function, which is applied after the weights and before the bias.
    pub activation: Box<dyn Fn(f32) -> f32>,
    /// The gradient of the activation function, used in backpropogation.
    pub activation_gradient: Box<dyn Fn(f32) -> f32>,
}

impl Activator {
    /// The equivalent matrix for the activation function for some set of weighted values.
    #[cfg(feature = "train")]
    pub fn activation_gradient_matrix<const OUTPUTS: usize>(
        &self,
        weighted: SVector<f32, OUTPUTS>,
    ) -> SMatrix<f32, OUTPUTS, OUTPUTS> {
        SMatrix::<f32, OUTPUTS, OUTPUTS>::from_fn(|i, j| {
            if i == j {
                (self.activation_gradient)(weighted[i])
            } else {
                0f32
            }
        })
    }
}
