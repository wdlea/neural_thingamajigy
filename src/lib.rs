mod calculus_shenanigans;
mod layer;
mod network;
mod train;
mod training_gradients;

pub use calculus_shenanigans::CalculusShenanigans;
pub use layer::Layer;
pub use network::Network;
pub use network::TrainingData;
pub use train::train;
pub use training_gradients::TrainingGradients;
