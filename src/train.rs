use nalgebra::SVector;

use crate::{network::NetworkData, Activator, Network};

/// Perform 1 training epoch on a network with training data.
/// `data` is a slice of `(INPUT, OUTPUT)` tuples.
pub fn train<
    'a,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>(
    data: &'a [(SVector<f32, INPUTS>, SVector<f32, OUTPUTS>)],
    network: &mut Network<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    learning_rate: f32,
    activator: &Activator,
) -> f32 {
    let mut loss = 0f32;

    let gradients: Vec<NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>> = data
        .iter()
        .map(
            #[expect(non_snake_case)]
            |(x, Y)| {
                let (predicted, training_data) = network.evaluate_training(*x, activator);

                let delta = Y - predicted;
                loss += delta.norm_squared();

                network.get_data(training_data, 2f32 * delta, activator) // squared error
            },
        )
        .collect();

    let gradient = NetworkData::mean(&gradients); // mean squared error

    network.apply_nudge(-&gradient, learning_rate); // minimise by going the other way

    loss / gradients.len() as f32
}
