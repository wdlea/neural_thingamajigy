extern crate std;

use super::{gradient::Gradient, SimpleNetwork};
use crate::{
    activators::Activator,
    layer::{Layer, LayerGradient},
    network::{RandomisableNetwork, TrainableNetwork},
};
use nalgebra::{RealField, SVector};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::{array::from_fn, iter::zip, mem::MaybeUninit};

/// Inputs to each layer of the network, used in training.
#[derive(Clone)]
pub struct TrainingInputs<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// The inputs to the network
    pub input: SVector<T, INPUTS>,
    /// The inputs to each hidden layer
    pub hidden_inputs: [SVector<T, WIDTH>; HIDDEN],
    /// The input to the last layer
    pub hidden_output: SVector<T, WIDTH>,
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > TrainableNetwork<T, INPUTS, OUTPUTS> for SimpleNetwork<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type LayerInputs = TrainingInputs<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    type Gradient = Gradient<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    /// The same thing as evaluate, but returns the inputs to each layer
    /// as well. This will incur a performance penalty.
    fn evaluate_training(
        &self,
        input: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> (
        SVector<T, OUTPUTS>,
        TrainingInputs<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) {
        let hidden_input = self.first.through(input, activator);

        let mut current_hidden = hidden_input;

        let hidden_inputs: [SVector<T, WIDTH>; HIDDEN] = from_fn(|i| {
            let layer_input = current_hidden;

            current_hidden = self.hidden[i].through(current_hidden, activator);

            layer_input
        });

        let output = self.last.through(current_hidden, activator);

        (
            output,
            TrainingInputs {
                input,
                hidden_inputs,
                hidden_output: current_hidden,
            },
        )
    }

    /// Computes NetworkData from TrainingInputs and the loss gradient of each output.
    fn get_gradient(
        &self,
        data: &TrainingInputs<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
        output_loss_gradients: SVector<T, OUTPUTS>,
        activator: &impl Activator<T>,
    ) -> (
        Gradient<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
        SVector<T, INPUTS>,
    ) {
        let (last, mut current_loss_gradient) =
            self.last
                .backpropogate(output_loss_gradients, data.hidden_output, activator);

        let mut hidden: [MaybeUninit<LayerGradient<T, WIDTH, WIDTH>>; HIDDEN] =
            from_fn(|_| MaybeUninit::uninit()); // every value gets something assigned to it eventually

        // This loop assigns something to every element of HIDDEN, so the uninitialized memory is eventually replaced with acceptable values
        for i in (0..HIDDEN).rev() {
            let param_gradients;

            (param_gradients, current_loss_gradient) = self.hidden[i].backpropogate(
                current_loss_gradient,
                data.hidden_inputs[i],
                activator,
            );

            hidden[i].write(param_gradients);
        }

        let (first, input_loss) =
            self.first
                .backpropogate(current_loss_gradient, data.input, activator);

        (
            Gradient {
                first,
                hidden: unsafe {
                    hidden
                        .as_ptr()
                        .cast::<[LayerGradient<T, WIDTH, WIDTH>; HIDDEN]>()
                        .read()
                }, // as MaybeUninit is a union: () | T, the largest type will be T and thus this will be correctly sized and i can do this cast
                last,
            },
            input_loss,
        )
    }

    /// Applies a nudge to the network
    fn apply_nudge(&mut self, nudge: Gradient<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>) {
        self.first
            .apply_shifts(nudge.first.weight_gradient, nudge.first.bias_gradient);

        for (layer, shift) in zip(&mut self.hidden, nudge.hidden) {
            layer.apply_shifts(shift.weight_gradient, shift.bias_gradient);
        }

        self.last
            .apply_shifts(nudge.last.weight_gradient, nudge.last.bias_gradient);
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > RandomisableNetwork<T> for SimpleNetwork<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
where
    Standard: Distribution<T>,
{
    /// Generates a random network with random((-1)-(1)) weights and biases.
    fn random(rng: &mut impl Rng) -> Self {
        Self {
            first: Layer::random(rng),
            hidden: from_fn(|_| Layer::random(rng)),
            last: Layer::random(rng),
        }
    }
}
