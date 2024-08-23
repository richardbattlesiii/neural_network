use ndarray::Array2;

pub const RELU:u8 = 0;
pub const SIGMOID:u8 = 1;
pub const TANH:u8 = 2;

pub fn activate(function_to_use: u8, input: &mut Array2<f32>) {
    match function_to_use {
        RELU => relu(input),
        SIGMOID => simgoid(input),
        TANH => tanh(input),
        _ => panic!("Invalid activation function.")
    }
}

fn relu(input: &mut Array2<f32>) {
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            if input[[row, col]] < 0.0 {
                input[[row, col]] *= 0.001;
            }
        }
    }
}

fn simgoid(input: &mut Array2<f32>) {
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            input[[row, col]] = 1.0/(1.0+(-input[[row, col]]).exp());
        }
    }
}

fn tanh(input: &mut Array2<f32>) {
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            input[[row, col]] = input[[row, col]].tanh();
        }
    }
}



pub fn activation_derivative(function_to_use: u8, input: &mut Array2<f32>){
    match function_to_use {
        RELU => relu_derivative(input),
        SIGMOID => simgoid_derivative(input),
        TANH => tanh_derivative(input),
        _ => panic!("Invalid actiavtion function.")
    }
}

fn relu_derivative(input: &mut Array2<f32>){
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            if input[[row, col]] < 0.0 {
                input[[row, col]] = 0.001;
            }
            else {
                input[[row, col]] = 1.0;
            }
        }
    }
}

fn simgoid_derivative(input: &mut Array2<f32>){
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            let sigmoid = 1.0/(1.0+(-input[[row, col]]).exp());
            input[[row, col]] = sigmoid*(1.0-sigmoid);
        }
    }
}

fn tanh_derivative(input: &mut Array2<f32>){
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            let tanh = input[[row, col]].tanh();
            input[[row, col]] = 1.0 - tanh*tanh;
        }
    }
}