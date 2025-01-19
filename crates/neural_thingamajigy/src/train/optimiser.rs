use crate::valueset::ValueSet;
use core::f32;
use nalgebra::RealField;

/// Represents an optimiser
pub trait Optimiser<T: RealField + Copy, G> {
    /// Transforms the gradient into the step to take
    fn transform(&mut self, gradient: &G) -> G;
}

/// The ADAM optimiser
pub struct AdamOptimiser<T: RealField + Copy, G: ValueSet<T>> {
    /// The momentum variable in the ADAM optimiser
    momentum: G,
    /// The velocity variable in the ADAM optimiser
    velocity: G,

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

impl<T: RealField + Copy, G: ValueSet<T> + Default> AdamOptimiser<T, G> {
    /// Creates a new AdamOptimiser with the given hyperparameters
    pub fn new(learning_rate: T, momentum_mixer: T, velocity_mixer: T) -> Self {
        Self {
            momentum: G::default(),
            velocity: G::default(),
            learning_rate,
            momentum_mixer,
            velocity_mixer,
            accumulated_momentum: momentum_mixer,
            accumulated_velocity: velocity_mixer,
        }
    }
}

impl<T: RealField + Copy + From<f32>, G: ValueSet<T> + Default> Default for AdamOptimiser<T, G> {
    fn default() -> Self {
        Self::new(0.001.into(), 0.9.into(), 0.999.into())
    }
}

impl<T: RealField + Copy, G: ValueSet<T>> Optimiser<T, G> for AdamOptimiser<T, G> {
    fn transform(&mut self, gradient: &G) -> G {
        // Calculate M[t+1] and V[t+1] respectively:
        self.momentum = self.momentum.binary_operation(gradient, |&mom, &gra| {
            mom * self.momentum_mixer + gra * (T::one() - self.momentum_mixer)
        });

        self.velocity = self.velocity.binary_operation(gradient, |&vel, &gra| {
            vel * self.velocity_mixer + gra.powi(2) * (T::one() - self.velocity_mixer)
        });

        // Calculate M^ [t+1] and V^ [t+1] respectively:
        let corrected_momentum = self
            .momentum
            .unary_operation(|&mom| mom / (T::one() - self.accumulated_momentum));
        let corrected_velocity = self
            .velocity
            .unary_operation(|&vel| vel / (T::one() - self.accumulated_velocity));

        // Update accumulated values
        self.accumulated_momentum *= self.momentum_mixer;
        self.accumulated_velocity *= self.velocity_mixer;

        corrected_momentum.binary_operation(&corrected_velocity, |&mom, &vel| {
            -self.learning_rate * mom
                / (vel.sqrt() + T::min_value().expect("T has no minimum value"))
        })
    }
}
