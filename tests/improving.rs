use nalgebra::{Vector1, Vector2};
use neural_thingamajigy::{
    activators, get_loss, loss::squared_error, optimiser::AdamOptimiser, train, SimpleNetwork,
};

/// The model should not get worse with training
#[test]
fn improvement_test() {
    let activator = activators::Sigmoid;

    let mut network = SimpleNetwork::<f32, 2, 1, 5, 2>::random();

    let data = [
        (Vector2::new(0f32, 0f32), Vector1::new(0f32)),
        (Vector2::new(1f32, 0f32), Vector1::new(1f32)),
        (Vector2::new(0f32, 1f32), Vector1::new(1f32)),
        (Vector2::new(1f32, 1f32), Vector1::new(0f32)),
    ];

    let mut opt = AdamOptimiser::default();

    let first_loss = get_loss(data.iter(), &mut network, &activator, &squared_error);

    for _ in 0..10 {
        train(
            data.iter(),
            &mut network,
            &activator,
            &squared_error,
            &mut opt,
        );
    }

    let last_loss = get_loss(data.iter(), &mut network, &activator, &squared_error);

    assert!(first_loss >= last_loss)
}
