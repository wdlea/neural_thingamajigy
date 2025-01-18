use std::ops::{Add, Mul, Neg, Sub};

use nalgebra::{RealField, SMatrix, SVector};

use crate::valueset::ValueSet;

/// Data about a layer generated via backpropogation used in training.
#[derive(Clone)]
pub struct LayerData<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> {
    /// The gradient of the weight values with respect to the loss function.
    pub weight_gradient: SMatrix<T, OUTPUTS, INPUTS>,

    /// The gradient of the bias values with respect to the loss function.
    pub bias_gradient: SVector<T, OUTPUTS>,
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> ValueSet<T>
    for LayerData<T, INPUTS, OUTPUTS>
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
    for LayerData<T, INPUTS, OUTPUTS>
{
    fn default() -> Self {
        Self {
            weight_gradient: SMatrix::zeros(),
            bias_gradient: SMatrix::zeros(),
        }
    }
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Add
    for &LayerData<T, INPUTS, OUTPUTS>
{
    type Output = LayerData<T, INPUTS, OUTPUTS>;

    fn add(self, rhs: Self) -> Self::Output {
        LayerData::<T, INPUTS, OUTPUTS> {
            weight_gradient: self.weight_gradient + rhs.weight_gradient,
            bias_gradient: self.bias_gradient + rhs.bias_gradient,
        }
    }
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Neg
    for &LayerData<T, INPUTS, OUTPUTS>
{
    type Output = LayerData<T, INPUTS, OUTPUTS>;

    fn neg(self) -> Self::Output {
        LayerData::<T, INPUTS, OUTPUTS> {
            weight_gradient: -self.weight_gradient,
            bias_gradient: -self.bias_gradient,
        }
    }
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Sub
    for &LayerData<T, INPUTS, OUTPUTS>
{
    type Output = LayerData<T, INPUTS, OUTPUTS>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<T: RealField + Copy, const INPUTS: usize, const OUTPUTS: usize> Mul<T>
    for &LayerData<T, INPUTS, OUTPUTS>
{
    type Output = LayerData<T, INPUTS, OUTPUTS>;

    fn mul(self, rhs: T) -> Self::Output {
        LayerData::<T, INPUTS, OUTPUTS> {
            weight_gradient: self.weight_gradient * rhs,
            bias_gradient: self.bias_gradient * rhs,
        }
    }
}
