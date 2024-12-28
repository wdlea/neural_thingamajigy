/// Defines LossFunction and some common instances
pub mod loss;

use loss::LossFunction;
use nalgebra::SVector;
use rand::Rng;

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
    learning_rate: f32,
    activator: &Activator,
    loss_function: &LossFunction<OUTPUTS>,
) -> f32 {
    let mut total_loss = 0f32;

    let gradients: Vec<NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>> = data
        .map(
            #[expect(non_snake_case)]
            |(x, Y)| {
                let (predicted, training_data) = network.evaluate_training(*x, activator);

                let (instance_loss, loss_gradient) = loss_function(Y, &predicted);
                total_loss += instance_loss;

                network.get_data(training_data, loss_gradient, activator) // squared error
            },
        )
        .collect();

    let gradient = NetworkData::mean(&gradients); // mean squared error

    network.apply_nudge(-&gradient, learning_rate); // minimise by going the other way

    total_loss / gradients.len() as f32
}

/// Utility for sampling data randomly
pub struct RandomSampler<'a, const INPUTS: usize, const OUTPUTS: usize, R: Rng> {
    /// A reference to the data to sample
    pub data: &'a [(SVector<f32, INPUTS>, SVector<f32, OUTPUTS>)],
    /// The RNG to use for sampling
    pub rng: &'a mut R,
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize, R: Rng> Iterator
    for RandomSampler<'a, INPUTS, OUTPUTS, R>
{
    type Item = &'a (SVector<f32, INPUTS>, SVector<f32, OUTPUTS>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.rng.gen::<usize>() % self.data.len();
        Some(&self.data[index])
    }
}
