/// Defines a simple network with limited customisation optionss
mod simple_network;

use rand::{distributions::Standard, prelude::Distribution, Rng};
pub use simple_network::SimpleNetwork;

use nalgebra::{RealField, SVector};

use crate::{activators::Activator, valueset::ValueSet};

/// Represents a neural network
pub trait Network<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    /// Evaluate the network with a set of inputs to return a set of outputs
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SVector<T, OUTPUTS>;
}

/// Represents a network that exposes training functionality
#[cfg(feature = "train")]
pub trait TrainableNetwork<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    /// The type representing the stored state of the network when given a particular inputs
    type LayerInputs;
    /// The type representing the gradient of the network with repect to (usually) the loss function
    type Gradient: ValueSet<T> + Default;

    /// Evaluates the network and returns the outputs and stored state for a given input
    fn evaluate_training(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> (SVector<T, OUTPUTS>, Self::LayerInputs);

    /// Backpropogates the network using the stored state to return the gradient with respect to the loss function
    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: SVector<T, OUTPUTS>,
        activator: &impl Activator<T>,
    ) -> (Self::Gradient, SVector<T, INPUTS>);

    /// Applies a nudge to the network
    fn apply_nudge(&mut self, nudge: Self::Gradient);
}

/// Represents a network that can be randomised
#[cfg(feature = "train")]
pub trait RandomisableNetwork<T>
where
    Standard: Distribution<T>,
{
    /// Generate a random network using rng
    fn random(rng: &mut impl Rng) -> Self;
}
