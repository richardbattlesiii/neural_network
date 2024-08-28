use ndarray::Array2;

pub const RELU:u8 = 0;
pub const SIGMOID:u8 = 1;
pub const TANH:u8 = 2;

pub fn activate(function_to_use: u8, input: f32) -> f32 {
    match function_to_use {
        RELU => relu(input),
        SIGMOID => sigmoid(input),
        TANH => tanh(input),
        _ => panic!("Invalid activation function.")
    }
}

pub fn activate_2d(function_to_use: u8, input: &mut Array2<f32>) {
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            match function_to_use {
                RELU => input[[row, col]] = relu(input[[row, col]]),
                SIGMOID => input[[row, col]] = sigmoid(input[[row, col]]),
                TANH => input[[row, col]] = tanh(input[[row, col]]),
                _ => panic!("Invalid activation function.")
            }
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



pub fn activation_derivative(function_to_use: u8, input: f32) -> f32 {
    match function_to_use {
        RELU => relu_derivative(input),
        SIGMOID => sigmoid_derivative(input),
        TANH => tanh_derivative(input),
        _ => panic!("Invalid activation function.")
    }
}

pub fn activation_derivative_2d(function_to_use: u8, input: &mut Array2<f32>) {
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            match function_to_use {
                RELU => input[[row, col]] = relu_derivative(input[[row, col]]),
                SIGMOID => input[[row, col]] = sigmoid_derivative(input[[row, col]]),
                TANH => input[[row, col]] = tanh_derivative(input[[row, col]]),
                _ => panic!("Invalid activation function.")
            }
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