use std::mem::MaybeUninit;

use nalgebra::{ComplexField, SMatrix};

pub trait ValueSet<T: Clone>: Sized {
    /// Executes f for every corresponding set of entries in gradients, returning the results
    fn nary_operation<const N: usize>(sets: [&Self; N], f: impl Fn([&T; N]) -> T) -> Self;

    /// Executes f for every entry, returning the transformed value
    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self {
        Self::nary_operation([&self], |[a]| f(a))
    }

    /// Executes f for every corresponding set of entries, returning the results
    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        Self::nary_operation([&self, other], |[a, b]| f(a, b))
    }

    /// Executes f for every corresponding set of entries in gradients
    fn nary_inspection<const N: usize>(sets: [&Self; N], f: &mut impl FnMut([&T; N]));

    /// Executes f for every entry
    fn unary_inspection(&self, f: &mut impl FnMut(&T)) {
        Self::nary_inspection([&self], &mut |[a]| f(a));
    }

    /// Executes f for every corresponding set of entries
    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T)) {
        Self::nary_inspection([&self, other], &mut |[a, b]| f(a, b));
    }

    fn all(v: T) -> Self;
}

impl<T: ComplexField, const WIDTH: usize, const HEIGHT: usize> ValueSet<T>
    for SMatrix<T, WIDTH, HEIGHT>
{
    fn nary_operation<const N: usize>(sets: [&Self; N], f: impl Fn([&T; N]) -> T) -> Self {
        let mut out = SMatrix::<T, WIDTH, HEIGHT>::zeros();

        for i in 0..(WIDTH * HEIGHT) {
            let mut arr = [MaybeUninit::<&T>::uninit(); N];
            for n in 0..N {
                arr[n].write(&sets[n][i]);
            }

            out[i] = f(unsafe {
                arr.map(|i| i.assume_init()) // I believe this is a no-op, so this will be optimised out
            });
        }
        out
    }

    fn nary_inspection<const N: usize>(sets: [&Self; N], f: &mut impl FnMut([&T; N])) {
        for i in 0..(WIDTH * HEIGHT) {
            let mut arr = [MaybeUninit::<&T>::uninit(); N];
            for n in 0..N {
                arr[n].write(&sets[n][i]);
            }

            f(unsafe {
                arr.map(|i| i.assume_init()) // I believe this is a no-op, so this will be optimised out
            });
        }
    }

    fn all(v: T) -> Self {
        Self::from_iterator([v].iter().cloned().cycle())
    }
}

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
