use std::f32::consts::E;

use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{train, Activator, Network};

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

    let mut network = Network::<2, 1, 2, 1>::random();

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut counter = 0;
    let learning_rate = -0.02f32; // - to minimse, + to maximise

    print!("epoch, loss, ");
    'training_loop: loop {
        println!("");
        print!("{}, ", counter);
        counter += 1;

        let mse = train(&data, &mut network, learning_rate, &activator);
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
