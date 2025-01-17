use nalgebra::RealField;
#[cfg(feature = "train")]
use nalgebra::{SMatrix, SVector};

/// Contains an activaton function and it's gradient
pub trait Activator<T: RealField + Copy> {
    /// The activation function, which is applied after the weights and before the bias.
    fn activation(&self, x: T) -> T;

    /// The gradient of the activation function, used in backpropogation.
    fn activation_gradient(&self, x: T) -> T;

    /// The equivalent matrix for the activation function for some set of weighted values.
    #[cfg(feature = "train")]
    fn activation_gradient_matrix<const OUTPUTS: usize>(
        &self,
        weighted: SVector<T, OUTPUTS>,
    ) -> SMatrix<T, OUTPUTS, OUTPUTS> {
        SMatrix::<T, OUTPUTS, OUTPUTS>::from_fn(|i, j| {
            if i == j {
                self.activation_gradient(weighted[i])
            } else {
                T::zero()
            }
        })
    }
}

/// The Sigmoid activation function
pub struct Sigmoid;

impl<T: RealField + Copy> Activator<T> for Sigmoid {
    fn activation(&self, x: T) -> T {
        T::one() / (T::one() + T::exp(x))
    }

    fn activation_gradient(&self, x: T) -> T {
        let sigma = self.activation(x);
        sigma * (T::one() - sigma)
    }
}

/// The Rectified Linear Unit activation function
pub struct Relu<T> {
    /// What gradient to have below zero
    pub leaky_gradient: T,
}

impl<T: RealField> Default for Relu<T> {
    fn default() -> Self {
        Self {
            leaky_gradient: T::zero(),
        }
    }
}

impl<T: RealField + Copy> Activator<T> for Relu<T> {
    fn activation(&self, x: T) -> T {
        if x >= T::zero(){
            x
        }else{
            self.leaky_gradient * x
        }
    }

    fn activation_gradient(&self, x: T) -> T {
        if x >= T::zero() {
            // in the rare case that x == 0, I think a gradient of 1 makes more sense
            T::one()
        } else {
            self.leaky_gradient
        }
    }
}

/// The Exponential Linear Unit activation function
pub struct Elu;

impl<T: RealField + Copy> Activator<T> for Elu {
    fn activation(&self, x: T) -> T {
        if x >= T::zero() {
            x
        } else {
            x.exp() - T::one()
        }
    }

    fn activation_gradient(&self, x: T) -> T {
        if x >= T::zero() {
            T::one()
        } else {
            x.exp()
        }
    }
}