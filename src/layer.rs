/// This defines the LayerData type, which contains data used in training about a layer.
#[cfg(feature = "train")]
mod layer_data;
#[cfg(feature = "train")]
pub use layer_data::LayerData;

/// Contains everything relating to training a layer.
#[cfg(feature = "train")]
mod layer_training;

use nalgebra::{SMatrix, SVector};

/// A layer of neurons in the network, this contains the weights, biases, activaiton function and it's gradient.
pub struct Layer<'a, const INPUTS: usize, const OUTPUTS: usize> {
    /// A matrix representing the weights of each value for each neuron.
    weight: SMatrix<f32, OUTPUTS, INPUTS>,
    /// The bias vector, which is added to each neuron after activation.
    bias: SVector<f32, OUTPUTS>,
    /// The activation function, which is applied after the weights and before the bias.
    activation: &'a dyn Fn(f32) -> f32,
    /// The gradient of the activation function, used in backpropogation.
    #[cfg(feature = "train")]
    activation_gradient: &'a dyn Fn(f32) -> f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Layer<'_, INPUTS, OUTPUTS> {
    /// Takes a set of inputs, and transforms them as per the weights, biases and activation function.
    pub fn through(&self, inputs: SVector<f32, INPUTS>) -> SVector<f32, OUTPUTS> {
        let weighted = self.weight * inputs;
        let activated =
            SVector::<f32, OUTPUTS>::from_iterator(weighted.iter().map(|v| (self.activation)(*v)));

        activated + self.bias
    }
}
