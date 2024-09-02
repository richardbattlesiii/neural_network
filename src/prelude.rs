pub use crate::layers::{
        layer::Layer,
        dense_layer::DenseLayer,
        convolutional_layer::*,
        softmax_layer::SoftmaxLayer,
        reshaping_layer::ReshapingLayer,
        dropout_layer::*,
        pooling_layer::*,
};
pub use crate::helpers::activation_functions::*;
pub use crate::networks::{
        neural_net::NeuralNet,
        dense_net,
        convolutional_net,
        gamer_net,
};