use nalgebra::{RealField, SVector};

use crate::{operations::normalize::TaxicabNormalize, ChainableNetwork, Network};

use super::Exp;

/// Performs the Softmax operation
pub struct Softmax;

impl<T: RealField + Copy, const INPUTS: usize> Network<T, INPUTS, INPUTS> for Softmax {
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        activator: &impl crate::activators::Activator<T>,
    ) -> SVector<T, INPUTS> {
        Exp.chain(TaxicabNormalize).evaluate(inputs, activator)
    }
}

#[cfg(feature = "train")]
use crate::{operations::Normalize, TrainableNetwork};

#[cfg(feature = "train")]
impl<T: RealField + Copy, const INPUTS: usize> TrainableNetwork<T, INPUTS, INPUTS> for Softmax {
    type LayerInputs = (SVector<T, INPUTS>, SVector<T, INPUTS>);

    type Gradient = ();

    fn evaluate_training(
        &self,
        inputs: nalgebra::SVector<T, INPUTS>,
        activator: &impl crate::activators::Activator<T>,
    ) -> (nalgebra::SVector<T, INPUTS>, Self::LayerInputs) {
        Exp.chain(Normalize).evaluate_training(inputs, activator)
    }

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: nalgebra::SVector<T, INPUTS>,
        activator: &impl crate::activators::Activator<T>,
    ) -> (Self::Gradient, nalgebra::SVector<T, INPUTS>) {
        (
            (),
            Exp.chain(Normalize)
                .get_gradient(layer_inputs, output_loss_gradients, activator)
                .1,
        )
    }

    fn apply_nudge(&mut self, _: Self::Gradient) {}
}

/// Tests
mod test {

    #[test]
    fn softmax_test() {
        use crate::{activators::Linear, operations::softmax::Softmax, Network};
        use nalgebra::Vector3;

        let softmaxed = Softmax.evaluate(Vector3::new(1f32, 2f32, 3f32), &Linear);
        assert!((softmaxed.norm() - 1f32) < 0.001f32);
        assert!((softmaxed - Vector3::new(0.090031f32, 0.244728f32, 0.665241f32)).norm() < 0.0001);
    }
}
