use nalgebra::SVector;
use std::{array::from_fn, iter::zip, mem::MaybeUninit};

use crate::{layer::Layer, layer::LayerData, network::network_data::NetworkData, Activator};

use super::Network;

/// Inputs to each layer of the network, used in training.
pub struct TrainingInputs<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// The inputs to the network
    pub input: SVector<f32, INPUTS>,
    /// The inputs to each hidden layer
    pub hidden_inputs: [SVector<f32, WIDTH>; HIDDEN],
    /// The input to the last layer
    pub hidden_output: SVector<f32, WIDTH>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Network<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// The same thing as evaluate, but returns the inputs to each layer
    /// as well. This will incur a performance penalty.
    pub fn evaluate_training(
        &self,
        input: SVector<f32, INPUTS>,
        activator: &Activator,
    ) -> (
        SVector<f32, OUTPUTS>,
        TrainingInputs<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) {
        let hidden_input = self.first.through(input, activator);

        let mut current_hidden = hidden_input;

        let hidden_inputs: [SVector<f32, WIDTH>; HIDDEN] = from_fn(|i| {
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
    pub fn get_data(
        &self,
        data: TrainingInputs<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
        output_loss_gradients: SVector<f32, OUTPUTS>,
        activator: &Activator,
    ) -> NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN> {
        let last = self
            .last
            .backpropogate(output_loss_gradients, data.hidden_output, activator);

        let mut hidden: [MaybeUninit<LayerData<WIDTH, WIDTH>>; HIDDEN] =
            from_fn(|_| MaybeUninit::uninit()); // every value gets something assigned to it eventually

        let mut current_loss_gradient = &last.loss_gradient;

        // This loop assigns something to every element of HIDDEN, so the uninitialized memory is eventually replaced with acceptable values
        for i in (0..HIDDEN).rev() {
            hidden[i].write(self.hidden[i].backpropogate(
                *current_loss_gradient,
                data.hidden_inputs[i],
                activator,
            ));

            // I am allowed to use hidden[i] as i just initialized it
            current_loss_gradient = unsafe { &hidden[i].assume_init_ref().loss_gradient };
            // this value just was assigned and nothing else will change it, thus it is safe to reference
        }

        let first = self
            .first
            .backpropogate(*current_loss_gradient, data.input, activator);

        NetworkData {
            first,
            hidden: unsafe {
                hidden
                    .as_ptr()
                    .cast::<[LayerData<WIDTH, WIDTH>; HIDDEN]>()
                    .read()
            }, // as MaybeUninit is a union: () | T, the largest type will be T and thus this will be correctly sized and i can do this cast
            last,
        }
    }

    /// Applies a nudge to the network, multiplied by the learning rate.
    pub fn apply_nudge(
        &mut self,
        nudge: NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) {
        self.first.apply_shifts(
            nudge.first.weight_gradient,
            nudge.first.bias_gradient,
        );

        for (layer, shift) in zip(&mut self.hidden, nudge.hidden) {
            layer.apply_shifts(
                shift.weight_gradient,
                shift.bias_gradient,
            );
        }

        self.last.apply_shifts(
            nudge.last.weight_gradient,
            nudge.last.bias_gradient,
        );
    }

    /// Generates a random network with random((-1)-(1)) weights and biases.
    pub fn random() -> Self {
        Self {
            first: Layer::random(),
            hidden: from_fn(|_| Layer::random()),
            last: Layer::random(),
        }
    }
}
