use super::matrix::Matrix;

pub const RELU:u8 = 0;
pub const SIGMOID:u8 = 1;
pub const TANH:u8 = 2;

pub fn activate(function_to_use: u8, input: &mut Matrix) {
    match function_to_use {
        RELU => relu(input),
        SIGMOID => simgoid(input),
        TANH => tanh(input),
        _ => panic!("Invalid actiavtion function.")
    }
}

fn relu(input: &mut Matrix) {
    for row in 0..input.rows {
        for col in 0..input.cols {
            let index = row*input.cols + col;
            if input.values[index] < 0.0 {
                input.values[index] = 0.001 * input.values[index];
            }
        }
    }
}

fn simgoid(input: &mut Matrix) {
    for row in 0..input.rows {
        for col in 0..input.cols {
            let index = row*input.cols + col;
            input.values[index] = 1.0/(1.0+(-input.values[index]).exp());
        }
    }
}

fn tanh(input: &mut Matrix) {
    for row in 0..input.rows {
        for col in 0..input.cols {
            let index = row*input.cols + col;
            input.values[index] = input.values[index].tanh();
        }
    }
}



pub fn activation_derivative(function_to_use: u8, input: &Matrix) -> Matrix {
    match function_to_use {
        RELU => return relu_derivative(input).copy(),
        SIGMOID => return simgoid_derivative(input).copy(),
        TANH => return tanh_derivative(input).copy(),
        _ => panic!("Invalid actiavtion function.")
    }
}

fn relu_derivative(input: &Matrix) -> Matrix {
    let mut output = input.copy();
    for row in 0..output.rows {
        for col in 0..output.cols {
            let index = row*output.cols + col;
            if output.values[index] < 0.0 {
                output.values[index] = 0.001;
            }
            else {
                output.values[index] = 1.0;
            }
        }
    }
    output
}

fn simgoid_derivative(input: &Matrix) -> Matrix {
    let mut output = input.copy();
    for row in 0..output.rows {
        for col in 0..output.cols {
            let index = row*output.cols+col;
            let sigmoid = 1.0/(1.0+(-output.values[index]).exp());
            output.values[index] = sigmoid*(1.0-sigmoid);
        }
    }
    output
}

fn tanh_derivative(input: &Matrix) -> Matrix {
    let mut output = input.copy();
    for row in 0..output.rows {
        for col in 0..output.cols {
            let index = row*output.cols+col;
            let tanh = output.values[index].tanh();
            output.values[index] = 1.0 - tanh*tanh;
        }
    }
    output
}