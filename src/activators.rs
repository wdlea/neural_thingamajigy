use std::f32::consts::E;

#[cfg(feature = "train")]
use nalgebra::{SMatrix, SVector};

/// Contains an activaton function and it's gradient
pub trait Activator {
    /// The activation function, which is applied after the weights and before the bias.
    fn activation(&self, x: f32) -> f32;

    /// The gradient of the activation function, used in backpropogation.
    fn activation_gradient(&self, x: f32) -> f32;

    /// The equivalent matrix for the activation function for some set of weighted values.
    #[cfg(feature = "train")]
    fn activation_gradient_matrix<const OUTPUTS: usize>(
        &self,
        weighted: SVector<f32, OUTPUTS>,
    ) -> SMatrix<f32, OUTPUTS, OUTPUTS> {
        SMatrix::<f32, OUTPUTS, OUTPUTS>::from_fn(|i, j| {
            if i == j {
                self.activation_gradient(weighted[i])
            } else {
                0f32
            }
        })
    }
}

/// The Sigmoid activation function
pub struct Sigmoid;

impl Activator for Sigmoid {
    fn activation(&self, x: f32) -> f32 {
        1f32 / (1f32 + E.powf(-x))
    }

    fn activation_gradient(&self, x: f32) -> f32 {
        let sigma = self.activation(x);
        sigma * (1f32 - sigma)
    }
}
