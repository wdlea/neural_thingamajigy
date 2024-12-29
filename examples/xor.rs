use std::f32::consts::E;

use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{
    loss::squared_error, optimiser::AdamOptimiser, train, Activator, Network,
};

fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + E.powf(-x))
}

fn dsigmoid_dx(x: f32) -> f32 {
    let sigma = sigmoid(x);

    sigma * (1f32 - sigma)
}

fn main() {
    let activator = Activator {
        activation: Box::new(sigmoid),
        activation_gradient: Box::new(dsigmoid_dx),
    };

    let mut network = Network::<2, 1, 5, 2>::random();

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut counter = 0;

    let mut opt = AdamOptimiser::new(0.001, 0.9, 0.999);

    print!("epoch, loss, ");
    'training_loop: loop {
        println!();
        print!("{}, ", counter);
        counter += 1;

        let mse = train(
            data.iter(),
            &mut network,
            &activator,
            &squared_error,
            &mut opt,
        );
        print!("{:.3}, ", mse);

        for (x, y) in data {
            let predicted = network.evaluate(x, &activator);

            let difference = (y - predicted).norm();
            if difference >= 0.5 {
                continue 'training_loop;
            }
        }

        break;
    }
}
