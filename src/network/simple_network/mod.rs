use crate::{activators::Activator, layer::Layer};
use nalgebra::{RealField, SVector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Network;

/// Defines Gradient, the object storing the gradient of all the weights and biases with respect to the loss function
mod gradient;
/// Defines the functions and objects related to training a network to extract and use a gradient
mod network_training;

/// Represents a network, a sequence of layer operations. Due to limitations in
/// the implementation, it must have at least 2 layers(first & last). The first
/// layer has `INPUTS` inputs and `WIDTH` outputs. All hidden layers take `WIDTH`
/// inputs and produce `WIDTH` outputs. The last layer has `WIDTH` inputs and
/// produces `OUTPUTS` outputs. There are `HIDDEN`(can be 0) hidden layers.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimpleNetwork<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// The first layer
    first: Layer<T, INPUTS, WIDTH>,
    /// All hidden layers
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    hidden: [Layer<T, WIDTH, WIDTH>; HIDDEN],
    /// The last layer
    last: Layer<T, WIDTH, OUTPUTS>,
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Network<T, INPUTS, OUTPUTS> for SimpleNetwork<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Evaluates a network given a set of inputs to produce it's outputs.
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SVector<T, OUTPUTS> {
        let hidden_inputs = self.first.through(inputs, activator);

        let mut current_hidden = hidden_inputs;
        for layer in &self.hidden {
            current_hidden = layer.through(current_hidden, activator);
        }

        self.last.through(current_hidden, activator)
    }
}
