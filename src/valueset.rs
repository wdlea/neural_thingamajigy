use nalgebra::{ComplexField, SMatrix};

pub trait ValueSet<T: Clone>: Sized {
    /// Executes f for every corresponding set of entries in gradients, returning the results
    fn nary_operation(sets: &[&Self], f: impl Fn(&[&T]) -> T) -> Self;

    /// Executes f for every entry, returning the transformed value
    fn unary_operation(&self, f: impl Fn(&T) -> T) -> Self {
        Self::nary_operation(&[&self], |a| f(a[0]))
    }

    /// Executes f for every corresponding set of entries, returning the results
    fn binary_operation(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        Self::nary_operation(&[&self, other], |a| f(a[0], a[1]))
    }

    /// Executes f for every corresponding set of entries in gradients
    fn nary_inspection(sets: &[&Self], f: &mut impl FnMut(&[&T]));

    /// Executes f for every entry
    fn unary_inspection(&self, f: &mut impl FnMut(&T)) {
        Self::nary_inspection(&[&self], &mut |a| f(a[0]));
    }

    /// Executes f for every corresponding set of entries
    fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&T, &T)) {
        Self::nary_inspection(&[&self, other], &mut |a| f(a[0], a[1]));
    }

    /// Creates a Self filled with v
    fn all(v: T) -> Self;
}

impl<T: ComplexField, const WIDTH: usize, const HEIGHT: usize> ValueSet<T>
    for SMatrix<T, WIDTH, HEIGHT>
{
    fn nary_operation(sets: &[&Self], f: impl Fn(&[&T]) -> T) -> Self {
        let mut out = SMatrix::<T, WIDTH, HEIGHT>::zeros();

        for i in 0..(WIDTH * HEIGHT) {
            let vec: Vec<_> = sets.iter().map(|&s| &s[i]).collect();

            out[i] = f(vec.as_slice());
        }
        out
    }

    fn nary_inspection(sets: &[&Self], f: &mut impl FnMut(&[&T])) {
        for i in 0..(WIDTH * HEIGHT) {
            let vec: Vec<_> = sets.iter().map(|&s| &s[i]).collect();

            f(vec.as_slice());
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
