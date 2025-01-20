use layer_chain::layer_chain;
use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{
    activators, loss::squared_error, optimiser::AdamOptimiser, train, Network, RandomisableNetwork,
};
use rand::rngs::OsRng;

layer_chain!(pub MyNetwork, f32, 2, 5, 5, 1);

fn main() {
    let activator = activators::Relu {
        leaky_gradient: 0.001,
    };

    let mut network = MyNetwork::random(&mut OsRng);

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut counter = 0;

    let mut opt = AdamOptimiser::default();

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
