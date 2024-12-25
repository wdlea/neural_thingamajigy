use std::{
    array::from_fn,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use crate::CalculusShenanigans;

pub struct TrainingGradients<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    pub first: CalculusShenanigans<INPUTS, WIDTH>,
    pub hidden: [CalculusShenanigans<WIDTH, WIDTH>; HIDDEN],
    pub last: CalculusShenanigans<WIDTH, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    pub fn mean(collection: &[Self]) -> Self {
        &collection.iter().sum::<Self>() * (1f32 / collection.len() as f32)
    }
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize>
    Sum<&'a TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>>
    for TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut sum = TrainingGradients::<INPUTS, OUTPUTS, WIDTH, HIDDEN>::default();

        for i in iter {
            sum = &sum + i;
        }

        sum
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Default
    for TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
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
    for &TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            first: &self.first + &rhs.first,
            hidden: from_fn(|i| &self.hidden[i] + &rhs.hidden[i]),
            last: &self.last + &rhs.last,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Neg
    for &TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn neg(self) -> Self::Output {
        Self::Output {
            first: -&self.first,
            hidden: from_fn(|i| -&self.hidden[i]),
            last: -&self.last,
        }
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Sub
    for &TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const WIDTH: usize, const HIDDEN: usize> Mul<f32>
    for &TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    type Output = TrainingGradients<INPUTS, OUTPUTS, WIDTH, HIDDEN>;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            first: &self.first * rhs,
            hidden: from_fn(|i| &self.hidden[i] * rhs),
            last: &self.last * rhs,
        }
    }
}
