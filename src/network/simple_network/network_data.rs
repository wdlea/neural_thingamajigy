use std::{
    array::from_fn,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use nalgebra::RealField;

use crate::{layer::LayerData, valueset::ValueSet};

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
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    /// Calculates the mean of a slice of network data
    /// Due to technical limitations, this collection length must be representable with T
    pub fn mean(collection: &[Self]) -> Self {
        let length = T::one() / T::from_usize(collection.len()).unwrap();
        &collection.iter().sum::<Self>() * (T::one() / length)
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > ValueSet<T> for NetworkData<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self {
        Self {
            first: self.first.unary_operation(&f),
            hidden: self.hidden.clone().map(|h| h.unary_operation(&f)),
            last: self.last.unary_operation(f),
        }
    }

    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        let mut hidden_counter = 0usize;
        Self {
            first: self.first.binary_operation(&other.first, &f),
            hidden: self.hidden.clone().map(|a| {
                let b = &other.hidden[hidden_counter];
                hidden_counter += 1;
                a.binary_operation(b, &f)
            }),
            last: self.last.binary_operation(&other.last, f),
        }
    }

    fn unary_inspection(&self, f: &mut impl FnMut(&T)) {
        self.first.unary_inspection(f);
        self.hidden.iter().for_each(|h| h.unary_inspection(f));
        self.last.unary_inspection(f);
    }

    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T)) {
        let mut hidden_counter = 0usize;

        self.first.binary_inspection(&other.first, f);
        self.hidden.iter().for_each(|a| {
            let b = &other.hidden[hidden_counter];
            hidden_counter += 1;
            a.binary_inspection(b, f)
        });
        self.last.binary_inspection(&other.last, f);
    }

    fn all(v: T) -> Self {
        let one_hidden = LayerData::all(v);
        Self {
            first: LayerData::all(v),
            hidden: from_fn(|_| one_hidden.clone()),
            last: LayerData::all(v),
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
