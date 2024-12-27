use std::f32::consts::E;

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
    let mut network = Network::<2, 1, 2, 1>::random(&sigmoid, &dsigmoid_dx);

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

        let mse = train(&data, &mut network, learning_rate);
        print!("{}, ", mse);
        
        for (x, y) in data {
            let predicted = network.evaluate(x);

            let difference = (y - predicted).norm();

            if difference >= 0.5 {
                continue 'training_loop;
            }
        }

        break;
    }
}
