use nalgebra::{SMatrix, SVector};

use super::{Activator, Layer, LayerData};

impl<const INPUTS: usize, const OUTPUTS: usize> Layer<INPUTS, OUTPUTS> {
    /// The gradient of the layer for a given set of inputs.
    pub fn gradient(
        &self,
        inputs: SVector<f32, INPUTS>,
        activator: &Activator,
    ) -> SMatrix<f32, OUTPUTS, INPUTS> {
        activator.activation_gradient_matrix(self.weight * inputs) * self.weight
    }

    /// Takes a set of inputs and loss gradients(with respect to the outputs) and calculates LayerData based on them.
    pub fn backpropogate(
        &self,
        loss_gradients: SVector<f32, OUTPUTS>,
        inputs: SVector<f32, INPUTS>,
        activator: &Activator,
    ) -> (LayerData<INPUTS, OUTPUTS>, SVector<f32, INPUTS>) {
        // each bias is shifted by it's respective loss
        let bias_gradient = loss_gradients;

        // each weight is shifted by it's pre-activation loss gradient multiplied by it's input
        let weight_gradient = activator.activation_gradient_matrix(self.weight * inputs)
            * loss_gradients
            * inputs.transpose();

        let gradient = self.gradient(inputs, activator);
        (
            LayerData {
                weight_gradient,
                bias_gradient,
            },
            gradient.transpose() * loss_gradients,
        )
    }

    /// Applies weight and bias shifts, normalized and multiplied by the learning rate.
    pub fn apply_shifts(
        &mut self,
        weight_direction: SMatrix<f32, OUTPUTS, INPUTS>,
        bias_direction: SVector<f32, OUTPUTS>,
    ) {
        self.weight += weight_direction;
        self.bias += bias_direction;
    }

    /// Generates a new layer with all values between -1 and 1
    pub fn random() -> Self {
        Self {
            weight: SMatrix::new_random() * 2f32 - SMatrix::from_fn(|_, _| 1f32),
            bias: SMatrix::new_random() * 2f32 - SMatrix::from_fn(|_, _| 1f32),
        }
    }
}
