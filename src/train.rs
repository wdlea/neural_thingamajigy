/// Defines LossFunction and some common instances
pub mod loss;

/// Defines Optimiser trait and ADAM
pub mod optimiser;

use std::iter::Sum;

use loss::LossFunction;
use nalgebra::{RealField, SVector};
use optimiser::Optimiser;

use crate::{activators::Activator, Network};
pub use crate::{layer::LayerData, network::NetworkData};

/// Perform 1 training epoch on a network with training data.
/// `data` is a slice of `(INPUT, OUTPUT)` tuples. This returns
/// the average loss of every sample as determined by
/// loss_function.
pub fn train<
    'a,
    T: RealField + Copy + From<u16>,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>(
    data: impl Iterator<Item = &'a (SVector<T, INPUTS>, SVector<T, OUTPUTS>)>,
    network: &mut Network<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    activator: &impl Activator<T>,
    loss_function: &LossFunction<T, OUTPUTS>,
    optimiser: &mut impl Optimiser<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
) -> T {
    let mut total_loss = T::zero();

    let gradients: Vec<NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>> = data
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

    total_loss / (gradients.len() as u16).into()
}

/// Calculates the average loss for a network from a set of data
pub fn get_loss<
    'a,
    T: RealField + Copy + Sum,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>(
    data: impl Iterator<Item = &'a (SVector<T, INPUTS>, SVector<T, OUTPUTS>)>,
    network: &Network<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    activator: &impl Activator<T>,
    loss_function: &LossFunction<T, OUTPUTS>,
) -> T {
    let mut counter = T::zero();
    data.inspect(|_| counter += T::one())
        .map(|(input, expected)| {
            let predicted = network.evaluate(*input, activator);

            loss_function(expected, &predicted).0
        })
        .sum::<T>()
        / counter
}
