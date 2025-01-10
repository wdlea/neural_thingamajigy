use core::f32;

use nalgebra::RealField;

use super::NetworkData;

/// Represents an optimiser
pub trait Optimiser<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>
{
    /// Transforms the gradient into the step to take
    fn transform(
        &mut self,
        gradient: &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) -> NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;
}

/// The ADAM optimiser
pub struct AdamOptimiser<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// The momentum variable in the ADAM optimiser
    momentum: NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    /// The velocity variable in the ADAM optimiser
    velocity: NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,

    /// Learning rate(alpha) hyperparameter
    pub learning_rate: T,
    /// Momentum "mixer" (beta1) hyperparameter
    pub momentum_mixer: T,
    /// Velocity "mixer" (beta2) hyperparameter
    pub velocity_mixer: T,

    /// beta1 ^ t field to optimise sequential generation
    accumulated_momentum: T,
    /// beta2 ^ t field to optimise sequential generation
    accumulated_velocity: T,
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > AdamOptimiser<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Creates a new AdamOptimiser with the given hyperparameters
    pub fn new(learning_rate: T, momentum_mixer: T, velocity_mixer: T) -> Self {
        Self {
            momentum: NetworkData::default(),
            velocity: NetworkData::default(),
            learning_rate,
            momentum_mixer,
            velocity_mixer,
            accumulated_momentum: momentum_mixer,
            accumulated_velocity: velocity_mixer,
        }
    }
}

impl<
        T: RealField + Copy + From<f32>,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Default for AdamOptimiser<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn default() -> Self {
        Self::new(0.001.into(), 0.9.into(), 0.999.into())
    }
}

impl<
        T: RealField + Copy + From<u8>,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Optimiser<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
    for AdamOptimiser<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn transform(
        &mut self,
        gradient: &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) -> NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN> {
        // Calculate M[t+1] and V[t+1] respectively:
        self.momentum = &(&self.momentum * self.momentum_mixer)
            + &(gradient * (T::one() - self.momentum_mixer));
        self.velocity = &(&self.velocity * self.velocity_mixer)
            + &(&gradient.map(&|v| v * v) * (T::one() - self.velocity_mixer));

        // Calculate M^ [t+1] and V^ [t+1] respectively:
        let corrected_momentum =
            &self.momentum * (T::one() / (T::one() - self.accumulated_momentum));
        let corrected_velocity =
            &self.velocity * (T::one() / (T::one() - self.accumulated_velocity));

        // Update accumulated values
        self.accumulated_momentum *= self.momentum_mixer;
        self.accumulated_velocity *= self.velocity_mixer;

        NetworkData::binary_elementwise(
            &corrected_momentum,
            &(&corrected_velocity.map(&|v: T| v.sqrt())
                + &NetworkData::all(T::min_value().unwrap())), // assuming min_value is equivialent to f32::EPSILON
            &|lhs, rhs| lhs / rhs * -self.learning_rate,
        )
    }
}
