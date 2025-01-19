/// This defines the LayerData type, which contains data used in training about a layer.
#[cfg(feature = "train")]
mod layer_data;
#[cfg(feature = "train")]
pub use layer_data::LayerGradient;

/// Contains everything relating to training a layer.
#[cfg(feature = "train")]
mod layer_training;

use nalgebra::{RealField, SMatrix, SVector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::activators::Activator;

/// A layer of neurons in the network, this contains the weights, biases, activaiton function and it's gradient.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Layer<T: RealField, const INPUTS: usize, const OUTPUTS: usize> {
    /// A matrix representing the weights of each value for each neuron.
    weight: SMatrix<T, OUTPUTS, INPUTS>,
    /// The bias vector, which is added to each neuron after activation.
    bias: SVector<T, OUTPUTS>,
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Layer<T, INPUTS, OUTPUTS> {
    /// Takes a set of inputs, and transforms them as per the weights, biases and activation function.
    pub fn through(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SVector<T, OUTPUTS> {
        let weighted = self.weight * inputs;
        let activated =
            SVector::<T, OUTPUTS>::from_iterator(weighted.iter().map(|v| activator.activation(*v)));

        activated + self.bias
    }
}
