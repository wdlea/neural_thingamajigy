use nalgebra::SVector;

/// A loss function, accepting the actual and predicted value (in that order) and providing the loss value & deriviative
pub type LossFunction<const N: usize> =
    dyn Fn(&SVector<f32, N>, &SVector<f32, N>) -> (f32, SVector<f32, N>);

/// The Squared Error loss function
pub fn squared_error<const N: usize>(
    actual: &SVector<f32, N>,
    predicted: &SVector<f32, N>,
) -> (f32, SVector<f32, N>) {
    let delta = actual - predicted;
    (delta.norm_squared(), 2f32 * delta)
}

/// The Absolute Error loss function
pub fn absoloute_error<const N: usize>(
    actual: &SVector<f32, N>,
    predicted: &SVector<f32, N>,
) -> (f32, SVector<f32, N>) {
    let delta = actual - predicted;
    (delta.norm(), delta.normalize())
}
