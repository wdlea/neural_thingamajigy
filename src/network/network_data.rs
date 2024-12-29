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

    /// Creates a NetworkData with all underlying values set to `value`
    pub fn all(value: f32) -> Self {
        Self {
            first: LayerData::all(value),
            hidden: from_fn(|_| LayerData::all(value)),
            last: LayerData::all(value),
        }
    }

    /// Transforms all underlying values using `f`
    #[must_use]
    pub fn map(&self, f: &impl Fn(f32) -> f32) -> Self {
        Self {
            first: self.first.map(f),
            hidden: from_fn(|i| self.hidden[i].map(f)),
            last: self.last.map(f),
        }
    }

    /// Performs f on corresponding entries in lhs and rhs to generate a new matrix with the results
    pub fn binary_elementwise(lhs: &Self, rhs: &Self, f: &impl Fn(f32, f32) -> f32) -> Self {
        Self {
            first: LayerData::binary_elementwise(&lhs.first, &rhs.first, f),
            hidden: from_fn(|i| LayerData::binary_elementwise(&lhs.hidden[i], &rhs.hidden[i], f)),
            last: LayerData::binary_elementwise(&lhs.last, &rhs.last, f),
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
