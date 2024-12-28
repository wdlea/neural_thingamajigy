use std::ops::{Add, Mul, Neg, Sub};

use nalgebra::{SMatrix, SVector, Scalar};

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

/// Performs the square root operation "element wise"
/// As described in: https://arxiv.org/pdf/1412.6980
fn matrix_component_sqrt<T: Scalar + Mul<T, Output = T> + Copy, const R: usize, const C: usize>(
    mat: SMatrix<T, R, C>,
) -> SMatrix<T, R, C> {
    SMatrix::<T, R, C>::from_fn(|i, j| mat[(i, j)] * mat[(i, j)])
}

impl<const INPUTS: usize, const OUTPUTS: usize> LayerData<INPUTS, OUTPUTS> {
    /// Performs the square operation "element wise"
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_square(&self) -> Self {
        Self {
            weight_gradient: self.weight_gradient.component_mul(&self.weight_gradient),
            bias_gradient: self.bias_gradient.component_mul(&self.bias_gradient),
            gradient: self.gradient.component_mul(&self.gradient),
            loss_gradient: self.loss_gradient.component_mul(&self.loss_gradient),
        }
    }

    /// Performs the square root operation "element wise"
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_sqrt(&self) -> Self {
        Self {
            weight_gradient: matrix_component_sqrt(self.weight_gradient),
            bias_gradient: matrix_component_sqrt(self.bias_gradient),
            gradient: matrix_component_sqrt(self.gradient),
            loss_gradient: matrix_component_sqrt(self.loss_gradient),
        }
    }

    /// Performs "element wise" division
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_div(&self, other: &Self) -> Self {
        Self {
            weight_gradient: self.weight_gradient.component_div(&other.weight_gradient),
            bias_gradient: self.bias_gradient.component_div(&other.bias_gradient),
            gradient: self.gradient.component_div(&other.gradient),
            loss_gradient: self.loss_gradient.component_div(&other.loss_gradient),
        }
    }

    /// Returns a LayerData with all values set to f32::EPSILON
    pub fn epsilon() -> Self {
        Self {
            weight_gradient: SMatrix::from_fn(|_, _| f32::EPSILON),
            bias_gradient: SMatrix::from_fn(|_, _| f32::EPSILON),
            gradient: SMatrix::from_fn(|_, _| f32::EPSILON),
            loss_gradient: SMatrix::from_fn(|_, _| f32::EPSILON),
        }
    }
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
