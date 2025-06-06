use nalgebra::{RealField, SVector};

use crate::Network;

/// Performs e^x for each value in the supplied vector
pub struct Exp;

impl<T: RealField + Copy, const INPUTS: usize> Network<T, INPUTS, INPUTS> for Exp {
    fn evaluate(
        &self,
        inputs: SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> SVector<T, INPUTS> {
        inputs.map(|i| i.exp())
    }
}

#[cfg(feature = "train")]
use crate::TrainableNetwork;

#[cfg(feature = "train")]
impl<T: RealField + Copy, const INPUTS: usize> TrainableNetwork<T, INPUTS, INPUTS> for Exp {
    type LayerInputs = SVector<T, INPUTS>;

    type Gradient = ();

    fn evaluate_training(
        &self,
        inputs: nalgebra::SVector<T, INPUTS>,
        activator: &impl crate::activators::Activator<T>,
    ) -> (nalgebra::SVector<T, INPUTS>, Self::LayerInputs) {
        let res = self.evaluate(inputs, activator);
        (res, res)
    }

    fn get_gradient(
        &self,
        layer_inputs: &Self::LayerInputs,
        output_loss_gradients: nalgebra::SVector<T, INPUTS>,
        _: &impl crate::activators::Activator<T>,
    ) -> (Self::Gradient, nalgebra::SVector<T, INPUTS>) {
        ((), layer_inputs.component_mul(&output_loss_gradients)) // chain rule
    }

    // noop
    fn apply_nudge(&mut self, _: Self::Gradient) {}
}

/// Tests
mod test {

    #[test]
    fn exp_test() {
        use super::Exp;
        use crate::{activators::Sigmoid, network::Network, TrainableNetwork};
        use core::f32::consts::E;
        use nalgebra::Vector1;

        let num = 3f32;
        let input = Vector1::new(num);
        let expected = num.exp();

        assert!(expected - E.powf(num) < 0.01);

        assert_eq!(expected, Exp.evaluate(input.clone(), &Sigmoid).x); // Sigmoid is arbitrary and shouldn't do anything

        let (answer, data) = Exp.evaluate_training(input.clone(), &Sigmoid);
        assert_eq!(expected, answer.x);

        let (_, grad) = Exp.get_gradient(&data, Vector1::new(1f32), &Sigmoid);
        assert_eq!(grad.x, expected); // As this is linear, the gradient should be the same as the value

        let mul = 5f32;
        let (_, grad) = Exp.get_gradient(&data, Vector1::new(mul), &Sigmoid);
        assert_eq!(grad.x, expected * mul); // Checking that get_gradient works for any loss gradient
    }
}
