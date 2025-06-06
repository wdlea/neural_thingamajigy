use nalgebra::{RealField, SVector};

use crate::Network;

/// Normalizes anything passed into it
pub struct Normalize;

impl<T: RealField + Copy, const INPUTS: usize> Network<T, INPUTS, INPUTS> for Normalize {
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> SVector<T, INPUTS> {
        inputs.normalize()
    }
}

#[cfg(feature = "train")]
use {crate::TrainableNetwork, nalgebra::SMatrix};

#[cfg(feature = "train")]
impl<T: RealField + Copy, const INPUTS: usize> TrainableNetwork<T, INPUTS, INPUTS> for Normalize {
    type LayerInputs = SVector<T, INPUTS>;

    type Gradient = ();

    fn evaluate_training(
        &self,
        inputs: nalgebra::SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> (nalgebra::SVector<T, INPUTS>, Self::LayerInputs) {
        (inputs.normalize(), inputs)
    }

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: nalgebra::SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> (Self::Gradient, nalgebra::SVector<T, INPUTS>) {
        let output = layer_inputs.normalize();

        (
            (),
            (SMatrix::identity() - output * output.transpose()) * output_loss_gradients
                / layer_inputs.norm(),
        )
    }

    // noop
    fn apply_nudge(&mut self, _: Self::Gradient) {}
}

/// Normalizes the inputs based on taxicab geometry
pub struct TaxicabNormalize;

impl<T: RealField + Copy, const INPUTS: usize> Network<T, INPUTS, INPUTS> for TaxicabNormalize {
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> SVector<T, INPUTS> {
        let sum = inputs.sum();

        inputs / sum
    }
}

#[cfg(feature = "train")]
impl<T: RealField + Copy, const INPUTS: usize> TrainableNetwork<T, INPUTS, INPUTS>
    for TaxicabNormalize
{
    type LayerInputs = SVector<T, INPUTS>;

    type Gradient = ();

    fn evaluate_training(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl crate::activators::Activator<T>,
    ) -> (SVector<T, INPUTS>, Self::LayerInputs) {
        (self.evaluate(inputs, activator), inputs)
    }

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> (Self::Gradient, SVector<T, INPUTS>) {
        let input_sum = layer_inputs.sum();
        let dot = layer_inputs.dot(&output_loss_gradients);

        (
            (),
            output_loss_gradients / input_sum - layer_inputs * dot / (input_sum.powi(2)),
        )
    }

    fn apply_nudge(&mut self, _: Self::Gradient) {}
}

/// Testing
mod test {
    #[test]
    fn test_normalize() {
        use core::f32::consts::PI;

        use nalgebra::Vector6;

        use crate::{
            activators::Sigmoid, operations::normalize::Normalize, Network, TrainableNetwork,
        };

        let vec = Vector6::new(1f32, 7f32, -4f32, PI, 100f32, -1700f32);
        let expected = vec.normalize();
        assert_eq!(Normalize.evaluate(vec, &Sigmoid), expected);
        assert!((expected.norm() - 1f32) < 0.001f32);

        let (out, inputs) = Normalize.evaluate_training(vec, &Sigmoid);
        assert_eq!(out, expected);

        // Just making sure it doesn't panic, not too sure how to check this is correct
        let (_, _) = Normalize.get_gradient(
            &inputs,
            Vector6::new(1f32, 1f32, 1f32, 1f32, 1f32, 1f32),
            &Sigmoid,
        );
    }
}
