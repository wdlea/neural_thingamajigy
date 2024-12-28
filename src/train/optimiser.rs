use core::f32;

use super::NetworkData;

/// Represents an optimiser
pub trait Optimiser<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
>
{
    /// Transforms the gradient into the step to take
    fn transform(
        &mut self,
        gradient: &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) -> NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>;
}

/// The ADAM optimiser
pub struct AdamOptimiser<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    momentum: NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    velocity: NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,

    learning_rate: f32,  // alpha
    momentum_mixer: f32, // beta[1]
    velocity_mixer: f32, // beta[2]

    accumulated_momentum: f32, // beta[1] ^ t
    accumulated_velocity: f32, // beta[2] ^ t
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    AdamOptimiser<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn new(learning_rate: f32, momentum_mixer: f32, velocity_mixer: f32) -> Self {
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

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Default
    for AdamOptimiser<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999)
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Optimiser<INPUTS, OUTPUTS, WIDTH, HIDDEN> for AdamOptimiser<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn transform(
        &mut self,
        gradient: &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>,
    ) -> NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN> {
        // Calculate M[t+1] and V[t+1] respectively:
        self.momentum =
            &(&self.momentum * self.momentum_mixer) + &(gradient * (1f32 - self.momentum_mixer));
        self.velocity = &(&self.velocity * self.velocity_mixer)
            + &(&gradient.element_square() * (1f32 - self.velocity_mixer));

        // Calculate M^ [t+1] and V^ [t+1] respectively:
        let corrected_momentum = &self.momentum * (1f32 / (1f32 - self.accumulated_momentum));
        let corrected_velocity = &self.velocity * (1f32 / (1f32 - self.accumulated_velocity));

        // Update accumulated values
        self.accumulated_momentum *= self.momentum_mixer;
        self.accumulated_velocity *= self.velocity_mixer;

        &corrected_momentum
            .element_div(&(&corrected_velocity.element_sqrt() + &NetworkData::epsilon()))
            * -self.learning_rate
    }
}
