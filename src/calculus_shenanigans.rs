use std::ops::{Add, Mul, Neg, Sub};

use nalgebra::{SMatrix, SVector};

pub struct CalculusShenanigans<const INPUTS: usize, const OUTPUTS: usize> {
    pub weight_shift: SMatrix<f32, OUTPUTS, INPUTS>,
    pub bias_shift: SVector<f32, OUTPUTS>,
    pub gradient: SMatrix<f32, OUTPUTS, INPUTS>,
    pub loss_gradient: SVector<f32, INPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Add for CalculusShenanigans<INPUTS, OUTPUTS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            weight_shift: self.weight_shift + rhs.weight_shift,
            bias_shift: self.bias_shift + rhs.bias_shift,
            gradient: self.gradient + rhs.gradient,
            loss_gradient: self.loss_gradient + rhs.loss_gradient,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Neg for CalculusShenanigans<INPUTS, OUTPUTS> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            weight_shift: -self.weight_shift,
            bias_shift: -self.bias_shift,
            gradient: -self.gradient,
            loss_gradient: -self.loss_gradient,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Sub for CalculusShenanigans<INPUTS, OUTPUTS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Mul<f32> for CalculusShenanigans<INPUTS, OUTPUTS> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            weight_shift: self.weight_shift * rhs,
            bias_shift: self.bias_shift * rhs,
            gradient: self.gradient * rhs,
            loss_gradient: self.loss_gradient * rhs,
        }
    }
}
