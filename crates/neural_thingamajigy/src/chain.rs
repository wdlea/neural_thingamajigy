use core::marker::PhantomData;

use nalgebra::{RealField, SVector};

use crate::{activators::Activator, Network};

/// 2 networks that have been chained together
pub struct ChainedNetwork<
    T: RealField + Copy,
    const INPUTS: usize,
    const MIDDLE: usize,
    const OUTPUTS: usize,
    A: Network<T, INPUTS, MIDDLE>,
    B: Network<T, MIDDLE, OUTPUTS>,
> {
    /// The first network to be evaluated
    pub first: A,
    /// The next network to be evaluated
    pub second: B,

    /// PhantomData to gaslight the compiler
    pub _marker: PhantomData<T>,
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const MIDDLE: usize,
        const OUTPUTS: usize,
        A: Network<T, INPUTS, MIDDLE>,
        B: Network<T, MIDDLE, OUTPUTS>,
    > Network<T, INPUTS, OUTPUTS> for ChainedNetwork<T, INPUTS, MIDDLE, OUTPUTS, A, B>
{
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SVector<T, OUTPUTS> {
        self.second
            .evaluate(self.first.evaluate(inputs, activator), activator)
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const MIDDLE: usize,
        const OUTPUTS: usize,
        A: Network<T, INPUTS, MIDDLE>,
        B: Network<T, MIDDLE, OUTPUTS>,
    > ChainedNetwork<T, INPUTS, MIDDLE, OUTPUTS, A, B>
{
    /// Chain several networks together
    pub fn chain(first: A, second: B) -> Self {
        ChainedNetwork {
            first,
            second,
            _marker: PhantomData,
        }
    }
}

/// A network that can be chained with others
pub trait ChainableNetwork<T: RealField + Copy, const INPUTS: usize, const MIDDLE: usize>
where
    Self: Network<T, INPUTS, MIDDLE> + Sized,
{
    /// Chain this network to next and return the chain
    /// ## Example
    /// ```rust
    ///     use neural_thingamajigy::{network, RandomisableNetwork, ChainableNetwork, Network, TrainableNetwork, activators::Relu};
    ///     use rand::rngs::OsRng;
    ///
    ///     network!(pub MyCoolNetwork, f32, 1, 2, 3);
    ///     network!(pub MyOtherNetwork, f32, 3, 2, 1);
    ///
    ///     let mut cool = MyCoolNetwork::random(&mut OsRng);
    ///     let mut other = MyOtherNetwork::random(&mut OsRng);
    ///     
    ///     // Passing mutable references to TrainableNetworks(which
    ///     // network! produces on builds with std support), means
    ///     // `chained` will also be a TrainiableNetwork.
    ///     let chained = (&mut cool).chain(&mut other);
    ///
    ///     let output1: nalgebra::Vector1<f32> = chained.evaluate(nalgebra::Vector1::<f32>::new(1f32), &Relu::default());
    ///     let (output2, _) = chained.evaluate_training(nalgebra::Vector1::<f32>::new(1f32), &Relu::default());
    ///
    ///     assert_eq!(output1, output2);
    /// ```
    fn chain<const OUTPUTS: usize, Next: Network<T, MIDDLE, OUTPUTS>>(
        self,
        next: Next,
    ) -> ChainedNetwork<T, INPUTS, MIDDLE, OUTPUTS, Self, Next> {
        ChainedNetwork {
            first: self,
            second: next,
            _marker: PhantomData,
        }
    }
}
impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const MIDDLE: usize,
        S: Network<T, INPUTS, MIDDLE> + Sized,
    > ChainableNetwork<T, INPUTS, MIDDLE> for S
{
}

#[cfg(feature = "train")]
use super::TrainableNetwork;
#[cfg(feature = "train")]
impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const MIDDLE: usize,
        const OUTPUTS: usize,
        A: Network<T, INPUTS, MIDDLE> + TrainableNetwork<T, INPUTS, MIDDLE>,
        B: Network<T, MIDDLE, OUTPUTS> + TrainableNetwork<T, MIDDLE, OUTPUTS>,
    > TrainableNetwork<T, INPUTS, OUTPUTS> for ChainedNetwork<T, INPUTS, MIDDLE, OUTPUTS, A, B>
{
    type LayerInputs = (A::LayerInputs, B::LayerInputs);

    type Gradient = (A::Gradient, B::Gradient);

    fn evaluate_training(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> (SVector<T, OUTPUTS>, Self::LayerInputs) {
        let (middle, first_data) = self.first.evaluate_training(inputs, activator);
        let (outputs, second_data) = self.second.evaluate_training(middle, activator);

        (outputs, (first_data, second_data))
    }

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: SVector<T, OUTPUTS>,
        activator: &impl Activator<T>,
    ) -> (Self::Gradient, SVector<T, INPUTS>) {
        let (second_gradient, middle_loss_gradient) =
            self.second
                .get_gradient(&layer_inputs.1, output_loss_gradients, activator);
        let (first_gradient, input_loss_gradient) =
            self.first
                .get_gradient(&layer_inputs.0, middle_loss_gradient, activator);

        ((first_gradient, second_gradient), input_loss_gradient)
    }

    fn apply_nudge(&mut self, nudge: Self::Gradient) {
        self.first.apply_nudge(nudge.0);
        self.second.apply_nudge(nudge.1);
    }
}
