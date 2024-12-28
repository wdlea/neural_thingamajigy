use std::f32::consts::E;

use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{loss::squared_error, optimiser::AdamOptimiser, train, Activator, Network};

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

    // Random Batching
    // let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut adam = AdamOptimiser::default();

    print!("epoch, loss, ");
    'training_loop: loop {
        println!();
        print!("{}, ", counter);
        counter += 1;

        // Random Batching
        // let mse = train(
        //     RandomSampler {
        //         data: &data,
        //         rng: &mut rng,
        //     }
        //     .take(4),
        //     &mut network,
        //     learning_rate,
        //     &activator,
        //     &squared_error,
        // );

        let mse = train(
            data.iter(),
            &mut network,
            &activator,
            &squared_error,
            &mut adam,
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
