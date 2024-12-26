use crate::CalculusShenanigans;
use nalgebra::{SMatrix, SVector};

pub struct Layer<'a, const INPUTS: usize, const OUTPUTS: usize> {
    weight: SMatrix<f32, OUTPUTS, INPUTS>,
    bias: SVector<f32, OUTPUTS>,
    activation: &'a dyn Fn(f32) -> f32,
    activation_gradient: &'a dyn Fn(f32) -> f32,
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize> Layer<'a, INPUTS, OUTPUTS> {
    pub fn through(&self, inputs: SVector<f32, INPUTS>) -> SVector<f32, OUTPUTS> {
        let weighted = self.weight * inputs;
        let activated =
            SVector::<f32, OUTPUTS>::from_iterator(weighted.iter().map(|v| (self.activation)(*v)));

        activated + self.bias
    }

    pub fn gradient(&self, inputs: SVector<f32, INPUTS>) -> SMatrix<f32, OUTPUTS, INPUTS> {
        self.activation_gradient_matrix(self.weight * inputs) * self.weight
    }

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

    pub fn backpropogate(
        &self,
        loss_gradients: SVector<f32, OUTPUTS>,
        inputs: SVector<f32, INPUTS>,
    ) -> CalculusShenanigans<INPUTS, OUTPUTS> {
        // each bias is shifted by it's respective loss
        let bias_shift = loss_gradients;

        // each weight is shifted by it's pre-activation loss gradient multiplied by it's input
        let weight_shift = self.activation_gradient_matrix(self.weight * inputs)
            * loss_gradients
            * inputs.transpose();

        let gradient = self.gradient(inputs);
        CalculusShenanigans {
            weight_shift,
            bias_shift,
            gradient,
            loss_gradient: gradient.transpose() * loss_gradients,
        }
    }

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

    pub fn zeroed(
        activation: &'a dyn Fn(f32) -> f32,
        activation_gradient: &'a dyn Fn(f32) -> f32,
    ) -> Self {
        Self {
            weight: SMatrix::zeros(),
            bias: SMatrix::zeros(),
            activation,
            activation_gradient,
        }
    }
}
