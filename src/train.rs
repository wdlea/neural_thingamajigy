/// Defines LossFunction and some common instances
pub mod loss;

/// Defines Optimiser trait and ADAM
pub mod optimiser;

use std::iter::Sum;

use loss::LossFunction;
use nalgebra::{RealField, SVector};
use optimiser::Optimiser;

pub use crate::layer::LayerData;
use crate::{
    activators::Activator,
    network::{Network, TrainableNetwork},
    valueset::mean,
    SimpleNetwork,
};

/// Perform 1 training epoch on a network with training data.
/// `data` is a slice of `(INPUT, OUTPUT)` tuples. This returns
/// the average loss of every sample as determined by
/// loss_function.
pub fn train<
    'a,
    T: RealField + Copy + From<u8>,
    N: TrainableNetwork<T, INPUTS, OUTPUTS>,
    const INPUTS: usize,
    const OUTPUTS: usize,
>(
    data: impl Iterator<Item = &'a (SVector<T, INPUTS>, SVector<T, OUTPUTS>)>,
    network: &mut N,
    activator: &impl Activator<T>,
    loss_function: &LossFunction<T, OUTPUTS>,
    optimiser: &mut impl Optimiser<T, N::Gradient>,
) -> T {
    let mut total_loss = T::zero();

    let gradients: Vec<N::Gradient> = data
        .map(
            #[expect(non_snake_case)]
            |(x, Y)| {
                let (predicted, training_data) = network.evaluate_training(*x, activator);

                let (instance_loss, loss_gradient) = loss_function(Y, &predicted);
                total_loss += instance_loss;

                network
                    .get_gradient(&training_data, loss_gradient, activator)
                    .0 // discard network input loss as this isn't deep learning
            },
        )
        .collect();

    let gradient = mean(gradients.as_slice()); // mean error

    let step = optimiser.transform(&gradient);

    network.apply_nudge(step);

    total_loss / (gradients.len() as u8).into()
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
    network: &SimpleNetwork<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
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
