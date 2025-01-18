use std::iter::zip;

use nalgebra::{ComplexField, SMatrix};

/// Represents anything which is a nested collection of a value
pub trait ValueSet<T: Clone>: Sized {
    /// Executes f for every entry, returning the transformed value
    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self;

    /// Executes f for every corresponding set of entries, returning the results
    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self;

    /// Executes f for every entry
    fn unary_inspection(&self, f: &mut impl FnMut(&T));

    /// Executes f for every corresponding set of entries
    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T));

    /// Creates a Self filled with v
    fn all(v: T) -> Self;
}

impl<T: ComplexField, const WIDTH: usize, const HEIGHT: usize> ValueSet<T>
    for SMatrix<T, WIDTH, HEIGHT>
{
    fn all(v: T) -> Self {
        Self::from_iterator([v].iter().cloned().cycle())
    }

    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self {
        self.map(|x| f(&x))
    }

    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        self.zip_map(other, |a, b| f(&a, &b))
    }

    fn unary_inspection(&self, f: &mut impl FnMut(&T)) {
        self.iter().for_each(f);
    }

    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T)) {
        zip(self.iter(), other.iter()).for_each(|(a, b)| f(a, b))
    }
}

/// Returns the sum and count of the ValueSets in v
pub fn sum_count<T: ComplexField + Copy, V: ValueSet<T> + Default>(v: &[V]) -> (V, T) {
    let mut count = T::zero();
    let sum = v.iter().fold(V::default(), |acc, ele| {
        count += T::one(); // assuming this is either saturating or panics on overflow
        acc.binary_operation(ele, |&a, &b| a + b)
    });

    (sum, count)
}

/// Returns the mean(component-wise) of the ValueSets in v
pub fn mean<T: ComplexField + Copy, V: ValueSet<T> + Default>(v: &[V]) -> V {
    let (sum, count) = sum_count(v);
    sum.unary_operation(|&x| x / count)
}

/// Tests for ValueSet on matrices
mod test {
    #[test]
    fn test_gradient_impl() {
        use nalgebra::SMatrix;

        use crate::valueset::ValueSet;
        let a = SMatrix::<f32, 8, 8>::new_random();
        let b = SMatrix::<f32, 8, 8>::new_random();

        assert_eq!(a + b, a.binary_operation(&b, |p, q| p + q))
    }
}
