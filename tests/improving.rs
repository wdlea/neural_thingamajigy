use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{
    activators, loss::squared_error, optimiser::AdamOptimiser, train, Network,
};

/// The model should not get worse with training
#[test]
fn improvement_test() {
    let activator = activators::Sigmoid;

    let mut network = Network::<2, 1, 5, 2>::random();

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut opt = AdamOptimiser::default();

    let first_loss = train(
        data.iter(),
        &mut network,
        &activator,
        &squared_error,
        &mut opt,
    );

    for _ in 0..10000 {
        train(
            data.iter(),
            &mut network,
            &activator,
            &squared_error,
            &mut opt,
        );
    }

    let last_loss = train(
        data.iter(),
        &mut network,
        &activator,
        &squared_error,
        &mut opt,
    );

    assert!(first_loss >= last_loss)
}
