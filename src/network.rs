/// Contains everything relating to training a network.
#[cfg(feature = "train")]
mod network_training;

/// This defines the NetworkData type, containing data used in training about an entire network.
#[cfg(feature = "train")]
mod network_data;
#[cfg(feature = "train")]
pub use network_data::NetworkData;

use nalgebra::SVector;

use crate::layer::Layer;

/// Represents a network, a sequence of layer operations. Due to limitations in
/// the implementation, it must have at least 2 layers(first & last). The first
/// layer has `INPUTS` inputs and `WIDTH` outputs. All hidden layers take `WIDTH`
/// inputs and produce `WIDTH` outputs. The last layer has `WIDTH` inputs and
/// produces `OUTPUTS` outputs. There are `HIDDEN`(can be 0) hidden layers.
pub struct Network<
    'a,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// The first layer
    first: Layer<'a, INPUTS, WIDTH>,
    /// All hidden layers
    hidden: [Layer<'a, WIDTH, WIDTH>; HIDDEN],
    /// The last layer
    last: Layer<'a, WIDTH, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Network<'_, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Evaluates a network given a set of inputs to produce it's outputs.
    pub fn evaluate(&self, inputs: SVector<f32, INPUTS>) -> SVector<f32, OUTPUTS> {
        let hidden_inputs = self.first.through(inputs);

        let mut current_hidden = hidden_inputs;
        for layer in &self.hidden {
            current_hidden = layer.through(current_hidden);
        }

        self.last.through(current_hidden)
    }
}
