use std::{array::from_fn, iter::zip, mem::MaybeUninit};

use nalgebra::SVector;

use crate::{CalculusShenanigans, Layer, TrainingGradients};

pub struct Network<
    'a,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    first: Layer<'a, INPUTS, WIDTH>,
    hidden: [Layer<'a, WIDTH, WIDTH>; HIDDEN],
    last: Layer<'a, WIDTH, OUTPUTS>,
}

pub struct TrainingData<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    input: SVector<f32, INPUTS>,
    hidden_inputs: [SVector<f32, WIDTH>; HIDDEN],
    hidden_output: SVector<f32, WIDTH>,
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Network<'a, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    pub fn evaluate(&self, inputs: SVector<f32, INPUTS>) -> SVector<f32, OUTPUTS> {
        let hidden_inputs = self.first.through(inputs);

        let mut current_hidden = hidden_inputs;
        for layer in &self.hidden {
            current_hidden = layer.through(current_hidden);
        }

        self.last.through(current_hidden)
    }

    pub fn evaluate_training(
        &self,
        input: SVector<f32, INPUTS>,
    ) -> (
        SVector<f32, OUTPUTS>,
        TrainingData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) {
        let hidden_input = self.first.through(input);

        let mut current_hidden = hidden_input;

        let hidden_inputs: [SVector<f32, WIDTH>; HIDDEN] = from_fn(|i| {
            let layer_input = current_hidden;

            current_hidden = self.hidden[i].through(current_hidden);

            layer_input
        });

        let output = self.last.through(current_hidden);

        (
            output,
            TrainingData {
                input,
                hidden_inputs,
                hidden_output: current_hidden,
            },
        )
    }

    pub fn get_gradients(
        &self,
        data: TrainingData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
        output_loss_gradients: SVector<f32, OUTPUTS>,
    ) -> TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN> {
        let last = self
            .last
            .backpropogate(output_loss_gradients, data.hidden_output);

        let mut hidden: [MaybeUninit<CalculusShenanigans<WIDTH, WIDTH>>; HIDDEN] =
            from_fn(|_| MaybeUninit::uninit()); // every value gets something assigned to it eventually

        let mut current_loss_gradient = &last.loss_gradient;

        let mut initialized: Vec<bool> = (0..HIDDEN).map(|_| false).collect();

        // This loop assigns something to every element of HIDDEN, so the uninitialized memory is eventually replaced with acceptable values
        for i in (0..HIDDEN).rev() {
            initialized[i] = true;
            hidden[i]
                .write(self.hidden[i].backpropogate(*current_loss_gradient, data.hidden_inputs[i]));

            // I am allowed to use hidden[i] as i just initialized it
            current_loss_gradient = unsafe { &hidden[i].assume_init_ref().loss_gradient };
            // this value just was assigned and nothing else will change it, thus it is safe to reference
        }

        assert!(initialized.iter().all(|v| *v));

        let first = self.first.backpropogate(*current_loss_gradient, data.input);

        TrainingGradients {
            first,
            hidden: unsafe {
                hidden
                    .as_ptr()
                    .cast::<[CalculusShenanigans<WIDTH, WIDTH>; HIDDEN]>()
                    .read()
            }, // as MaybeUninit is a union: () | T, the largest type will be T and thus this will be correctly sized and i can do this cast
            last,
        }
    }

    pub fn apply_nudge(
        &mut self,
        nudge: TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
        learning_rate: f32,
    ) {
        self.first.apply_shifts(
            nudge.first.weight_shift,
            nudge.first.bias_shift,
            learning_rate,
        );

        for (layer, shift) in zip(&mut self.hidden, nudge.hidden) {
            layer.apply_shifts(shift.weight_shift, shift.bias_shift, learning_rate);
        }

        self.last.apply_shifts(
            nudge.last.weight_shift,
            nudge.last.bias_shift,
            learning_rate,
        );
    }

    pub fn random(
        activation: &'a dyn Fn(f32) -> f32,
        activation_gradient: &'a dyn Fn(f32) -> f32,
    ) -> Self {
        Self {
            first: Layer::random(activation, activation_gradient),
            hidden: from_fn(|_| Layer::random(activation, activation_gradient)),
            last: Layer::random(activation, activation_gradient),
        }
    }
}
