use std::ops::{Add, Mul, Neg, Sub};

use nalgebra::{SMatrix, SVector};

/// Data about a layer generated via backpropogation used in training.
pub struct LayerData<const INPUTS: usize, const OUTPUTS: usize> {
    /// The gradient of the weight values with respect to the loss function.
    pub weight_gradient: SMatrix<f32, OUTPUTS, INPUTS>,

    /// The gradient of the bias values with respect to the loss function.
    pub bias_gradient: SVector<f32, OUTPUTS>,

    /// The gradient of the inputs with respect to the outputs.
    pub gradient: SMatrix<f32, OUTPUTS, INPUTS>,

    /// The gradient of the inputs with respect to the loss function.
    pub loss_gradient: SVector<f32, INPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Default for LayerData<INPUTS, OUTPUTS> {
    fn default() -> Self {
        Self {
            weight_gradient: SMatrix::zeros(),
            bias_gradient: SMatrix::zeros(),
            gradient: SMatrix::zeros(),
            loss_gradient: SMatrix::zeros(),
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Add for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn add(self, rhs: Self) -> Self::Output {
        LayerData::<INPUTS, OUTPUTS> {
            weight_gradient: self.weight_gradient + rhs.weight_gradient,
            bias_gradient: self.bias_gradient + rhs.bias_gradient,
            gradient: self.gradient + rhs.gradient,
            loss_gradient: self.loss_gradient + rhs.loss_gradient,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Neg for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn neg(self) -> Self::Output {
        LayerData::<INPUTS, OUTPUTS> {
            weight_gradient: -self.weight_gradient,
            bias_gradient: -self.bias_gradient,
            gradient: -self.gradient,
            loss_gradient: -self.loss_gradient,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Sub for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Mul<f32> for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn mul(self, rhs: f32) -> Self::Output {
        LayerData::<INPUTS, OUTPUTS> {
            weight_gradient: self.weight_gradient * rhs,
            bias_gradient: self.bias_gradient * rhs,
            gradient: self.gradient * rhs,
            loss_gradient: self.loss_gradient * rhs,
        }
    }
}
