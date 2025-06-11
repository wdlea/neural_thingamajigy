//! This crate defines a generic feedforward neural network featuring custom activation functions and training using the mean-squared-error.
#![deny(missing_docs, clippy::missing_docs_in_private_items)]
#![no_std]

/// Defines the Activator type
pub mod activators;
/// Defines the ChainedNetwork type and chain, supporting joining networks together
mod chain;
/// This defines the Layer type, representing a layer of neurons and handles weighting, activation and biases.
mod layer;
/// This defines a network type, containing a sequence of layers.
mod network;
/// Defines common operations for [pre/post]processing
pub mod operations;
/// This holds the train function, allowing users to train their networks via MSE.
#[cfg(feature = "train")]
mod train;
/// Defines the ValueSet trait, which abstracts over anything which is a nested collection of a value
#[cfg(feature = "train")]
pub mod valueset;

pub use {chain::ChainableNetwork, layer::Layer, network::*, network_macro::network};
#[cfg(feature = "train")]
pub use {train::*, valueset::ValueSet};
