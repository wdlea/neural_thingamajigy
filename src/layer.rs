/// This defines the LayerData type, which contains data used in training about a layer.
#[cfg(feature = "train")]
mod layer_data;
#[cfg(feature = "train")]
pub use layer_data::LayerData;

/// Contains everything relating to training a layer.
#[cfg(feature = "train")]
mod layer_training;

use nalgebra::{SMatrix, SVector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Activator;

/// A layer of neurons in the network, this contains the weights, biases, activaiton function and it's gradient.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Layer<const INPUTS: usize, const OUTPUTS: usize> {
    /// A matrix representing the weights of each value for each neuron.
    weight: SMatrix<f32, OUTPUTS, INPUTS>,
    /// The bias vector, which is added to each neuron after activation.
    bias: SVector<f32, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Layer<INPUTS, OUTPUTS> {
    /// Takes a set of inputs, and transforms them as per the weights, biases and activation function.
    pub fn through(
        &self,
        inputs: SVector<f32, INPUTS>,
        activator: &Activator,
    ) -> SVector<f32, OUTPUTS> {
        let weighted = self.weight * inputs;
        let activated = SVector::<f32, OUTPUTS>::from_iterator(
            weighted.iter().map(|v| (activator.activation)(*v)),
        );

        activated + self.bias
    }
}
