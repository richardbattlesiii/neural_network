use ndarray::ArrayD;

pub const LINEAR:u8 = 0;
pub const RELU:u8 = 1;
pub const SIGMOID:u8 = 2;
pub const TANH:u8 = 3;

pub fn activate(function_to_use: u8, input: &mut ArrayD<f32>) {
    for value in input.iter_mut() {
        match function_to_use {
            LINEAR => {}, //Does nothing to the input
            RELU => *value = relu(*value),
            SIGMOID => *value = sigmoid(*value),
            TANH => *value = tanh(*value),
            _ => panic!("Invalid activation function.")
        }
    }
}

fn relu(input: f32) -> f32 {
    if input < 0.0 {
        0.001*input
    }
    else {
        input
    }
}

fn sigmoid(input: f32) -> f32 {
    1.0/(1.0+(-input).exp())
}

fn tanh(input: f32) -> f32 {
    input.tanh()
}



pub fn activation_derivative(function_to_use: u8, input: &mut ArrayD<f32>) {
    for value in input.iter_mut() {
        match function_to_use {
            LINEAR => {} //Does nothing to the input.
            RELU => *value = relu_derivative(*value),
            SIGMOID => *value = sigmoid_derivative(*value),
            TANH => *value = tanh_derivative(*value),
            _ => panic!("Invalid activation function.")
        }
    }
}

fn relu_derivative(input: f32) -> f32 {
    if input < 0.0 {
        0.001
    }
    else {
        1.0
    }
}

fn sigmoid_derivative(input: f32) -> f32 {
    let sigmoid = sigmoid(input);
    sigmoid*(1.0-sigmoid)
}

fn tanh_derivative(input: f32) -> f32 {
    let tanh = tanh(input);
    1.0 - tanh*tanh
}