use std::{
    array::from_fn,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use crate::layer::LayerData;

/// Data about a network generated via backpropogation used in training.
pub struct NetworkData<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// Data about the first layer
    pub first: LayerData<INPUTS, WIDTH>,
    /// Data about all the hidden layers
    pub hidden: [LayerData<WIDTH, WIDTH>; HIDDEN],
    /// Data about the last layer
    pub last: LayerData<WIDTH, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Calculates the mean of a slice of network data
    pub fn mean(collection: &[Self]) -> Self {
        &collection.iter().sum::<Self>() * (1f32 / collection.len() as f32)
    }

    /// Performs the square operation "element wise"
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_square(&self) -> Self {
        Self {
            first: self.first.element_square(),
            hidden: from_fn(|i| self.hidden[i].element_square()),
            last: self.last.element_square(),
        }
    }

    /// Performs the square root operation "element wise"
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_sqrt(&self) -> Self {
        Self {
            first: self.first.element_sqrt(),
            hidden: from_fn(|i| self.hidden[i].element_sqrt()),
            last: self.last.element_sqrt(),
        }
    }

    /// Performs "element wise" division
    /// As described in: https://arxiv.org/pdf/1412.6980
    pub fn element_div(&self, other: &Self) -> Self {
        Self {
            first: self.first.element_div(&other.first),
            hidden: from_fn(|i| self.hidden[i].element_div(&other.hidden[i])),
            last: self.last.element_div(&other.last),
        }
    }

    /// Returns a NetworkData with all values set to f32::EPSILON
    pub fn epsilon() -> Self {
        Self {
            first: LayerData::epsilon(),
            hidden: from_fn(|_| LayerData::epsilon()),
            last: LayerData::epsilon(),
        }
    }
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Sum<&'a NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>>
    for NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut sum = NetworkData::<INPUTS, OUTPUTS, WIDTH, HIDDEN>::default();

        for i in iter {
            sum = &sum + i;
        }

        sum
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Default
    for NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn default() -> Self {
        Self {
            first: Default::default(),
            hidden: from_fn(|_| Default::default()),
            last: Default::default(),
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Add
    for &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            first: &self.first + &rhs.first,
            hidden: from_fn(|i| &self.hidden[i] + &rhs.hidden[i]),
            last: &self.last + &rhs.last,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Neg
    for &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn neg(self) -> Self::Output {
        Self::Output {
            first: -&self.first,
            hidden: from_fn(|i| -&self.hidden[i]),
            last: -&self.last,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Sub
    for &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Mul<f32>
    for &NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            first: &self.first * rhs,
            hidden: from_fn(|i| &self.hidden[i] * rhs),
            last: &self.last * rhs,
        }
    }
}
