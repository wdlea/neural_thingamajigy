use std::ops::{Add, Mul, Neg, Sub};

use nalgebra::{SMatrix, SVector, Scalar};

/// Data about a layer generated via backpropogation used in training.
#[derive(Clone)]
pub struct LayerData<const INPUTS: usize, const OUTPUTS: usize> {
    /// The gradient of the weight values with respect to the loss function.
    pub weight_gradient: SMatrix<f32, OUTPUTS, INPUTS>,

    /// The gradient of the bias values with respect to the loss function.
    pub bias_gradient: SVector<f32, OUTPUTS>,
}

/// Performs f(lhs, rhs) on every value to generate a new Matrix
fn mat_binary_elementwise<T: Copy, O: Scalar, const R: usize, const C: usize>(
    lhs: &SMatrix<T, R, C>,
    rhs: &SMatrix<T, R, C>,
    f: &impl Fn(T, T) -> O,
) -> SMatrix<O, R, C> {
    SMatrix::from_fn(|i, j| f(lhs[(i, j)], rhs[(i, j)]))
}

impl<const INPUTS: usize, const OUTPUTS: usize> LayerData<INPUTS, OUTPUTS> {
    /// Creates a LayerData with all values(within *all* matrices) set to `value`
    pub fn all(value: f32) -> Self {
        Self {
            weight_gradient: SMatrix::from_fn(|_, _| value),
            bias_gradient: SMatrix::from_fn(|_, _| value),
        }
    }

    /// Performs .map(f) on *all* matrices contained by LayerData
    #[must_use]
    pub fn map(&self, f: &impl Fn(f32) -> f32) -> Self {
        Self {
            weight_gradient: self.weight_gradient.map(f),
            bias_gradient: self.bias_gradient.map(f),
        }
    }

    /// Performs f(lhs, rhs) on every value to generate a new LayerData
    #[must_use]
    pub fn binary_elementwise(lhs: &Self, rhs: &Self, f: &impl Fn(f32, f32) -> f32) -> Self {
        Self {
            weight_gradient: mat_binary_elementwise(&lhs.weight_gradient, &rhs.weight_gradient, f),
            bias_gradient: mat_binary_elementwise(&lhs.bias_gradient, &rhs.bias_gradient, f),
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Default for LayerData<INPUTS, OUTPUTS> {
    fn default() -> Self {
        Self {
            weight_gradient: SMatrix::zeros(),
            bias_gradient: SMatrix::zeros(),
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Add for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn add(self, rhs: Self) -> Self::Output {
        LayerData::<INPUTS, OUTPUTS> {
            weight_gradient: self.weight_gradient + rhs.weight_gradient,
            bias_gradient: self.bias_gradient + rhs.bias_gradient,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Neg for &LayerData<INPUTS, OUTPUTS> {
    type Output = LayerData<INPUTS, OUTPUTS>;

    fn neg(self) -> Self::Output {
        LayerData::<INPUTS, OUTPUTS> {
            weight_gradient: -self.weight_gradient,
            bias_gradient: -self.bias_gradient,
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
        }
    }
}
