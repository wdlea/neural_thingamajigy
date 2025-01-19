use nalgebra::{RealField, SVector};

/// A loss function, accepting the actual and predicted value (in that order) and providing the loss value & deriviative
pub type LossFunction<T, const N: usize> =
    dyn Fn(&SVector<T, N>, &SVector<T, N>) -> (T, SVector<T, N>);

/// The Squared Error loss function
pub fn squared_error<T: RealField + Copy, const N: usize>(
    actual: &SVector<T, N>,
    predicted: &SVector<T, N>,
) -> (T, SVector<T, N>) {
    let delta = predicted - actual;
    (delta.norm_squared(), delta + delta) // there is no longer a definition of 2, so delta + delta is a workaround for 2 * delta
}

/// The Absolute Error loss function
pub fn absoloute_error<T: RealField, const N: usize>(
    actual: &SVector<T, N>,
    predicted: &SVector<T, N>,
) -> (T, SVector<T, N>) {
    let delta = predicted - actual;
    (delta.norm(), delta.normalize())
}
