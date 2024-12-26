use nalgebra::SVector;

use crate::{Network, TrainingGradients};

pub fn train<
    'a,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>(
    data: &'a [(SVector<f32, INPUTS>, SVector<f32, OUTPUTS>)],
    mut network: Network<'a, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    learning_rate: f32,
) -> Network<'a, INPUTS, OUTPUTS, WIDTH, HIDDEN> {
    let gradients: Vec<TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>> = data
        .iter()
        .map(
            #[expect(non_snake_case)]
            |(x, Y)| {
                let (predicted, training_data) = network.evaluate_training(*x);

                let delta = Y - predicted;

                network.get_gradients(training_data, 2f32 * delta) // squared error
            },
        )
        .collect();

    let gradient = TrainingGradients::mean(&gradients); // mean squared error

    network.apply_nudge(-&gradient, learning_rate); // minimise by going the other way

    network
}
