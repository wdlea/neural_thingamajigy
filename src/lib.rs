//! This crate defines a generic feedforward neural network featuring custom activation functions and training using the mean-squared-error.

#![deny(missing_docs, clippy::missing_docs_in_private_items)]

/// This defines the LayerData type, which contains data used in training about a layer.
mod layer_data;

/// This defines the Layer type, representing a layer of neurons and handles weighting, activation and biases.
mod layer;

/// This defines a network type, containing a sequence of layers.
mod network;

/// This holds the train function, allowing users to train their networks via MSE.
mod train;

/// This defines the NetworkData type, containing data used in training about an entire network.
mod network_data;

pub use layer::Layer;
pub use layer_data::LayerData;
pub use network::Network;
pub use network::TrainingInputs;
pub use network_data::NetworkData;
pub use train::train;
