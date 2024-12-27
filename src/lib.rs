//! This crate defines a generic feedforward neural network featuring custom activation functions and training using the mean-squared-error.

#![deny(missing_docs, clippy::missing_docs_in_private_items)]

/// This defines the Layer type, representing a layer of neurons and handles weighting, activation and biases.
mod layer;

/// This defines a network type, containing a sequence of layers.
mod network;

/// This holds the train function, allowing users to train their networks via MSE.
#[cfg(feature = "train")]
mod train;

pub use network::Network;

#[cfg(feature = "train")]
pub use train::train;
