extern crate std;

use crate::valueset::ValueSet;
use nalgebra::{RealField, SMatrix, SVector};

/// Data about a layer generated via backpropogation used in training.
#[derive(Clone)]
pub struct LayerGradient<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    /// The gradient of the weight values with respect to the loss function.
    pub weight_gradient: SMatrix<T, OUTPUTS, INPUTS>,

    /// The gradient of the bias values with respect to the loss function.
    pub bias_gradient: SVector<T, OUTPUTS>,
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> ValueSet<T>
    for LayerGradient<T, INPUTS, OUTPUTS>
{
    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self {
        Self {
            weight_gradient: self.weight_gradient.unary_operation(&f),
            bias_gradient: self.bias_gradient.unary_operation(&f),
        }
    }

    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        Self {
            weight_gradient: self
                .weight_gradient
                .binary_operation(&other.weight_gradient, &f),
            bias_gradient: self
                .bias_gradient
                .binary_operation(&other.bias_gradient, &f),
        }
    }

    fn unary_inspection(&self, f: &mut impl FnMut(&T)) {
        self.weight_gradient.unary_inspection(f);
        self.bias_gradient.unary_inspection(f);
    }

    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T)) {
        self.weight_gradient
            .binary_inspection(&other.weight_gradient, f);
        self.bias_gradient
            .binary_inspection(&other.bias_gradient, f);
    }

    fn all(v: T) -> Self {
        Self {
            weight_gradient: SMatrix::all(v),
            bias_gradient: SMatrix::all(v),
        }
    }
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Default
    for LayerGradient<T, INPUTS, OUTPUTS>
{
    fn default() -> Self {
        Self {
            weight_gradient: SMatrix::zeros(),
            bias_gradient: SMatrix::zeros(),
        }
    }
}
