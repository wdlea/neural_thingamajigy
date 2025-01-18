//! This crate defines a generic feedforward neural network featuring custom activation functions and training using the mean-squared-error.
#![deny(missing_docs, clippy::missing_docs_in_private_items)]

/// Defines the Activator type
pub mod activators;
/// This defines the Layer type, representing a layer of neurons and handles weighting, activation and biases.
mod layer;
/// This defines a network type, containing a sequence of layers.
mod network;
/// This holds the train function, allowing users to train their networks via MSE.
#[cfg(feature = "train")]
mod train;
#[cfg(feature = "train")]
mod valueset;

pub use network::SimpleNetwork;
#[cfg(feature = "train")]
pub use train::*;
