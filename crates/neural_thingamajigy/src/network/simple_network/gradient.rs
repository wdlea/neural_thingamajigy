extern crate std;

use crate::{layer::LayerGradient, valueset::ValueSet};
use nalgebra::RealField;
use std::array::from_fn;
/// Data about a network generated via backpropogation used in training.
#[derive(Clone)]
pub struct Gradient<
    T: RealField + Copy,
    const INPUTS: usize,
    const OUTPUTS: usize,
    const WIDTH: usize,
    const HIDDEN: usize,
> {
    /// Data about the first layer
    pub first: LayerGradient<T, INPUTS, WIDTH>,
    /// Data about all the hidden layers
    pub hidden: [LayerGradient<T, WIDTH, WIDTH>; HIDDEN],
    /// Data about the last layer
    pub last: LayerGradient<T, WIDTH, OUTPUTS>,
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > ValueSet<T> for Gradient<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
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
        let one_hidden = LayerGradient::all(v);
        Self {
            first: LayerGradient::all(v),
            hidden: from_fn(|_| one_hidden.clone()),
            last: LayerGradient::all(v),
        }
    }
}

impl<
        T: RealField + Copy,
        const INPUTS: usize,
        const OUTPUTS: usize,
        const WIDTH: usize,
        const HIDDEN: usize,
    > Default for Gradient<T, INPUTS, OUTPUTS, WIDTH, HIDDEN>
{
    fn default() -> Self {
        Self {
            first: Default::default(),
            hidden: from_fn(|_| Default::default()),
            last: Default::default(),
        }
    }
}
