/// Defines LossFunction and some common instances
pub mod loss;

/// Defines Optimiser trait and ADAM
pub mod optimiser;

use loss::LossFunction;
use nalgebra::SVector;
use optimiser::Optimiser;

pub use crate::{layer::LayerData, network::NetworkData};
use crate::{Activator, Network};

/// Perform 1 training epoch on a network with training data.
/// `data` is a slice of `(INPUT, OUTPUT)` tuples. This returns
/// the average loss of every sample as determined by
/// loss_function.
pub fn train<
    'a,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>(
    data: impl Iterator<Item = &'a (SVector<f32, INPUTS>, SVector<f32, OUTPUTS>)>,
    network: &mut Network<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    activator: &Activator,
    loss_function: &LossFunction<OUTPUTS>,
    optimiser: &mut impl Optimiser<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
) -> f32 {
    let mut total_loss = 0f32;

    let gradients: Vec<NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>> = data
        .map(
            #[expect(non_snake_case)]
            |(x, Y)| {
                let (predicted, training_data) = network.evaluate_training(*x, activator);

                let (instance_loss, loss_gradient) = loss_function(Y, &predicted);
                total_loss += instance_loss;

                network.get_data(training_data, loss_gradient, activator).0 // discard network input loss as this isn't deep learning
            },
        )
        .collect();

    let gradient = NetworkData::mean(&gradients); // mean error

    let step = optimiser.transform(&gradient);

    network.apply_nudge(step);

    total_loss / gradients.len() as f32
}
