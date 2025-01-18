use super::{Layer, LayerGradient};
use crate::activators::Activator;
use nalgebra::{RealField, SMatrix, SVector};
use rand::{distributions::Standard, prelude::Distribution};

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Layer<T, INPUTS, OUTPUTS> {
    /// The gradient of the layer for a given set of inputs.
    pub fn gradient(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> SMatrix<T, OUTPUTS, INPUTS> {
        activator.activation_gradient_matrix(self.weight * inputs) * self.weight
    }

    /// Takes a set of inputs and loss gradients(with respect to the outputs) and calculates LayerData based on them.
    pub fn backpropogate(
        &self,
        loss_gradients: SVector<T, OUTPUTS>,
        inputs: SVector<T, INPUTS>,
        activator: &impl Activator<T>,
    ) -> (LayerGradient<T, INPUTS, OUTPUTS>, SVector<T, INPUTS>) {
        // each bias is shifted by it's respective loss
        let bias_gradient = loss_gradients;

        // each weight is shifted by it's pre-activation loss gradient multiplied by it's input
        let weight_gradient = activator.activation_gradient_matrix(self.weight * inputs)
            * loss_gradients
            * inputs.transpose();

        let gradient = self.gradient(inputs, activator);
        (
            LayerGradient {
                weight_gradient,
                bias_gradient,
            },
            gradient.transpose() * loss_gradients,
        )
    }

    /// Applies weight and bias shifts, normalized and multiplied by the learning rate.
    pub fn apply_shifts(
        &mut self,
        weight_direction: SMatrix<T, OUTPUTS, INPUTS>,
        bias_direction: SVector<T, OUTPUTS>,
    ) {
        self.weight += weight_direction;
        self.bias += bias_direction;
    }

    /// Generates a new layer with all values between -1 and 1
    pub fn random() -> Self
    where
        Standard: Distribution<T>,
    {
        let two = T::one() + T::one();
        Self {
            weight: SMatrix::new_random() * two - SMatrix::from_fn(|_, _| T::one()),
            bias: SMatrix::new_random() * two - SMatrix::from_fn(|_, _| T::one()),
        }
    }
}
