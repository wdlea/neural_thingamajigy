use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{
    activators, loss::squared_error, network, optimiser::AdamOptimiser, train, Network,
    RandomisableNetwork,
};
use rand::rngs::OsRng;

// Create a neural network called MyNetwork(public) using f32s with widths 2, 5, 5, 1.
network!(pub MyNetwork, f32, true, 2, 5, 5, 1);

fn main() {
    // Create the activation function
    let activator = activators::Relu {
        leaky_gradient: 0.001,
    };

    // Make an instance with weights and biases randomly intialized
    let mut network = MyNetwork::random(&mut OsRng);
    // Training data in (input, expected output) pairs
    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    // Epoch counter
    let mut counter = 0;

    // Create a new instance of the ADAM optimser
    let mut opt = AdamOptimiser::default();

    // CSV column headings
    print!("epoch, loss, ");
    'training_loop: loop {
        println!();
        print!("{}, ", counter);
        counter += 1;

        // Perform 1 iteration of training on the network with the data
        let mse = train(
            data.iter(),
            &mut network,
            &activator,
            &squared_error, // Use the [mean] squared error activation function.
            &mut opt,
        );
        print!("{:.3}, ", mse);

        for (x, y) in data {
            // Use the network to make a prediction on a piece of data
            let predicted = network.evaluate(x, &activator);

            let difference = (y - predicted).norm();
            // If the prediction isn't close enough, continue training
            if difference >= 0.5 {
                continue 'training_loop;
            }
        }
        // If all predictions were close enough to target values, stop training, we are done
        break;
    }
}
