use nalgebra::{SMatrix, SVector};

use super::{Layer, LayerData};

impl<'a, const INPUTS: usize, const OUTPUTS: usize> Layer<'a, INPUTS, OUTPUTS> {
    /// The gradient of the layer for a given set of inputs.
    pub fn gradient(&self, inputs: SVector<f32, INPUTS>) -> SMatrix<f32, OUTPUTS, INPUTS> {
        self.activation_gradient_matrix(self.weight * inputs) * self.weight
    }

    /// The equivalent matrix for the activation function for some set of weighted values.
    fn activation_gradient_matrix(
        &self,
        weighted: SVector<f32, OUTPUTS>,
    ) -> SMatrix<f32, OUTPUTS, OUTPUTS> {
        SMatrix::<f32, OUTPUTS, OUTPUTS>::from_fn(|i, j| {
            if i == j {
                (self.activation_gradient)(weighted[i])
            } else {
                0f32
            }
        })
    }

    /// Takes a set of inputs and loss gradients(with respect to the outputs) and calculates LayerData based on them.
    pub fn backpropogate(
        &self,
        loss_gradients: SVector<f32, OUTPUTS>,
        inputs: SVector<f32, INPUTS>,
    ) -> LayerData<INPUTS, OUTPUTS> {
        // each bias is shifted by it's respective loss
        let bias_shift = loss_gradients;

        // each weight is shifted by it's pre-activation loss gradient multiplied by it's input
        let weight_shift = self.activation_gradient_matrix(self.weight * inputs)
            * loss_gradients
            * inputs.transpose();

        let gradient = self.gradient(inputs);
        LayerData {
            weight_gradient: weight_shift,
            bias_gradient: bias_shift,
            gradient,
            loss_gradient: gradient.transpose() * loss_gradients,
        }
    }

    /// Applies weight and bias shifts, normalized and multiplied by the learning rate.
    pub fn apply_shifts(
        &mut self,
        weight_direction: SMatrix<f32, OUTPUTS, INPUTS>,
        bias_direction: SVector<f32, OUTPUTS>,
        learning_rate: f32,
    ) {
        //only shift values if there is a shift, otherwise normalize causes NaN due to division by 0 magnitude
        if weight_direction.norm() > 0f32 {
            let weight_shift = weight_direction.normalize() * learning_rate;
            self.weight += weight_shift;
        }

        if bias_direction.norm() > 0f32 {
            let bias_shift = bias_direction.normalize() * learning_rate;
            self.bias += bias_shift;
        }
    }

    /// Generates a new layer with all values between 0 and 1.
    pub fn random(
        activation: &'a dyn Fn(f32) -> f32,
        activation_gradient: &'a dyn Fn(f32) -> f32,
    ) -> Self {
        Self {
            weight: SMatrix::new_random(),
            bias: SMatrix::new_random(),
            activation,
            activation_gradient,
        }
    }
}
