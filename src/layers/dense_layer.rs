use std::any::TypeId;
use std::os::windows;

use crate::helpers::activation_functions::*;
use rand::Rng;
use std::fmt;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct DenseLayer {
    input_size:usize,
    output_size:usize,

    pub learning_rate:f32,
    pub activation_function:u8,

    pub weights:Array2<f32>,
    pub biases:Array1<f32>
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, learning_rate: f32, activation_function: u8) -> DenseLayer {
        DenseLayer {
            input_size,
            output_size,
            learning_rate,
            activation_function,
            weights: Array2::zeros((input_size, output_size)),
            biases: Array1::zeros(output_size)
        }
    }
    pub fn initialize(&mut self){
        let xavier = f32::sqrt(6.0/((self.input_size + self.output_size) as f32 ));
        self.weights = Array2::random((self.input_size, self.output_size), Uniform::new(-xavier, xavier));
        self.biases = Array1::random(self.output_size, Uniform::new(0.0, 0.001));
    }

    pub fn pass(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut product = input.dot(&self.weights);
        let biases_reshaped = self.biases.clone().insert_axis(ndarray::Axis(0));
        product += &biases_reshaped;
        activate(self.activation_function, &mut product);
        product
    }

    pub fn backpropagate(&mut self, input: &Array2<f32>, my_output: &Array2<f32>, error: &Array2<f32>) -> Array2<f32> {
        
        let mut dl_da = my_output.clone();
        activation_derivative(self.activation_function, &mut dl_da);
        dl_da *= error;

        let weight_gradients = input.t().dot(&dl_da);
        let bias_gradients = error.sum_axis(ndarray::Axis(0));
        let output = error.dot(&self.weights.t());
        self.biases.scaled_add(-self.learning_rate, &bias_gradients);
        self.weights.scaled_add(-self.learning_rate, &weight_gradients);

        output
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }

    pub fn set_weights(&mut self, weights: &Array2<f32>) {
        self.weights = weights.clone();
    }

    pub fn set_biases(&mut self, biases: &Array1<f32>) {
        self.biases = biases.clone();
    }

    pub fn get_parameters(&self) -> (&Array2<f32>, &Array1<f32>) {
        (&self.weights, &self.biases)
    }

    pub fn add_noise(&mut self, range: f32) {
        let mut rng = rand::thread_rng();
        for node in 0..self.output_size {
            for input_node in 0..self.input_size {
                self.weights[[input_node, node]] *= (rng.gen::<f32>() - 0.5) * range + 1.0;
            }
            self.biases[node] *= (rng.gen::<f32>() - 0.5) * range + 1.0;
        }
    }

    pub fn write_params_to_string(&self) -> String {
        let mut output = "".to_string();
        for i in 0..self.input_size {
            for j in 0..self.output_size {
                output += &(self.weights[[i, j]].to_string()+" ")
            }
        }
        output += "\n";
        for i in 0..self.input_size {
            output += &(self.biases[i].to_string()+" ")
        }
        output
    }
}

impl fmt::Display for DenseLayer{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Weights:\n{}Biases:\n{}", self.weights, self.biases)?;
        Ok(())
    }
}

impl Clone for DenseLayer {
    fn clone(&self) -> DenseLayer {
        DenseLayer {
            input_size: self.input_size,
            output_size: self.output_size,
            learning_rate: self.learning_rate,
            activation_function: self.activation_function,
            weights: self.weights.clone(),
            biases: self.biases.clone()
        }
    }
}