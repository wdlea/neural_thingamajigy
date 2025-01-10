use std::{
    array::from_fn,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use nalgebra::RealField;

use crate::layer::LayerData;

/// Data about a network generated via backpropogation used in training.
#[derive(Clone)]
pub struct NetworkData<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// Data about the first layer
    pub first: LayerData<T, INPUTS, WIDTH>,
    /// Data about all the hidden layers
    pub hidden: [LayerData<T, WIDTH, WIDTH>; HIDDEN],
    /// Data about the last layer
    pub last: LayerData<T, WIDTH, OUTPUTS>,
}

impl<
        T: RealField + Copy + From<u8>,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Calculates the mean of a slice of network data
    /// Due to technical limitations, this collection cannot be longer than u8::MAX (255 elements)
    pub fn mean(collection: &[Self]) -> Self {
        assert!(collection.len() <= u8::MAX.into());
        &collection.iter().sum::<Self>() * (T::one() / (collection.len() as u8).into())
    }

    /// Creates a NetworkData with all underlying values set to `value`
    pub fn all(value: T) -> Self {
        Self {
            first: LayerData::all(value),
            hidden: from_fn(|_| LayerData::all(value)),
            last: LayerData::all(value),
        }
    }

    /// Transforms all underlying values using `f`
    #[must_use]
    pub fn map(&self, f: &impl Fn(T) -> T) -> Self {
        Self {
            first: self.first.map(f),
            hidden: from_fn(|i| self.hidden[i].map(f)),
            last: self.last.map(f),
        }
    }

    /// Performs f on corresponding entries in lhs and rhs to generate a new matrix with the results
    pub fn binary_elementwise(lhs: &Self, rhs: &Self, f: &impl Fn(T, T) -> T) -> Self {
        Self {
            first: LayerData::binary_elementwise(&lhs.first, &rhs.first, f),
            hidden: from_fn(|i| LayerData::binary_elementwise(&lhs.hidden[i], &rhs.hidden[i], f)),
            last: LayerData::binary_elementwise(&lhs.last, &rhs.last, f),
        }
    }
}

impl<
        'a,
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Sum<&'a NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>>
    for NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut sum = NetworkData::<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>::default();

        for i in iter {
            sum = &sum + i;
        }

        sum
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Default for NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn default() -> Self {
        Self {
            first: Default::default(),
            hidden: from_fn(|_| Default::default()),
            last: Default::default(),
        }
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Add for &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            first: &self.first + &rhs.first,
            hidden: from_fn(|i| &self.hidden[i] + &rhs.hidden[i]),
            last: &self.last + &rhs.last,
        }
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Neg for &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn neg(self) -> Self::Output {
        Self::Output {
            first: -&self.first,
            hidden: from_fn(|i| -&self.hidden[i]),
            last: -&self.last,
        }
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Sub for &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Mul<T> for &NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            first: &self.first * rhs,
            hidden: from_fn(|i| &self.hidden[i] * rhs),
            last: &self.last * rhs,
        }
    }
}
