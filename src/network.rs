mod simple_network;
use rand::{distributions::Standard, prelude::Distribution, Rng};
pub use simple_network::SimpleNetwork;

use nalgebra::{RealField, SVector};

use crate::activators::Activator;

pub trait Network<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SVector<T, OUTPUTS>;
}

#[cfg(feature = "train")]
pub trait TrainableNetwork<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    type LayerInputs;
    type Gradient;

    fn evaluate_training(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> (SVector<T, OUTPUTS>, Self::LayerInputs);

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: SVector<T, OUTPUTS>,
        activator: &impl Activator<T>,
    ) -> (Self::Gradient, SVector<T, INPUTS>);

    fn apply_nudge(&mut self, nudge: Self::Gradient);
}

#[cfg(feature = "train")]
pub trait RandomisableNetwork<T>
where
    Standard: Distribution<T>,
{
    fn random(rng: &mut impl Rng) -> Self;
}
