use std::f32::consts::E ;

use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{train, Network};

fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + E.powf(-x))
}

fn dsigmoid_dx(x: f32) -> f32 {
    let sigma = sigmoid(x);

    sigma * (1f32 - sigma)
}

fn main() {
    let mut network = Network::<2, 1, 2, 1>::zeroed(&sigmoid, &dsigmoid_dx);

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut counter = 0;

    'training_loop: loop {
        println!("Epoch #{}", counter);
        counter += 1;

        train(&data, &mut network, 0f32);

        for (x, y) in data{
            let predicted = network.evaluate(x);
            println!("Actual: {}, Predicted: {}", y, predicted);

            let difference = (y - predicted).norm();

            println!("Difference: {}", difference);

            if difference >= 0.5{
                continue 'training_loop;
            }
        }

        break;
    }

    println!("Something working was generated, yay!");
}
