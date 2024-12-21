use nalgebra::{SMatrix, SVector};

pub struct Layer<'a, const INPUTS: usize, const OUTPUTS: usize> {
    weight: SMatrix<f32, OUTPUTS, INPUTS>,
    bias: SVector<f32, OUTPUTS>,
    activation: &'a dyn Fn(f32) -> f32,
    activation_gradient: &'a dyn Fn(f32) -> f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Layer<'_, INPUTS, OUTPUTS> {
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

    pub fn backpropogate(&self, loss_gradients: SVector<f32, OUTPUTS>, inputs: SVector<f32, INPUTS>) -> (SMatrix<f32, OUTPUTS, INPUTS>, SVector<f32, OUTPUTS>)  {
        // each bias is shifted by it's respective loss
        let bias_shift = loss_gradients;

        // each weight is shifted by it's pre-activation loss gradient multiplied by it's input
        let weight_shift = self.activation_gradient_matrix(self.weight * inputs) * loss_gradients * inputs.transpose();

        (weight_shift, bias_shift)
    }
}
