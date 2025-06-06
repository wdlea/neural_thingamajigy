/// Defines the Exp struct which represents the Exp operation
mod exp;
/// Defines the Normalize struct which normalizes a vector passed into it
mod normalize;
/// Defines the Softmax struct which performs Softmax
mod softmax;

pub use {exp::Exp, normalize::Normalize, softmax::Softmax};
