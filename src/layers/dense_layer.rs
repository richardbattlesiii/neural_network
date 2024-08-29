use crate::helpers::activation_functions::*;
use rand::Rng;
use std::fmt;
use ndarray::{Array1, ArrayView1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct DenseLayer {
    input_size:usize,
    output_size:usize,

    learning_rate:f32,
    activation_function:u8,
    lambda:f32,

    weights:Array2<f32>,
    biases:Array1<f32>
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize,
            learning_rate: f32, lambda: f32, activation_function: u8)
            -> DenseLayer {
        DenseLayer {
            input_size,
            output_size,
            learning_rate,
            activation_function,
            lambda,
            weights: Array2::zeros((input_size, output_size)),
            biases: Array1::zeros(output_size)
        }
    }
    
    pub fn initialize(&mut self){
        let xavier = f32::sqrt(6.0/((self.input_size + self.output_size) as f32 ));
        self.weights = Array2::random((self.input_size, self.output_size), Uniform::new(-xavier, xavier));
        self.biases = Array1::random(self.output_size, Uniform::new(-0.01, 0.01));
    }

    pub fn pass(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        // println!("Input:\n{}", input);
        let mut product = input.dot(&self.weights);
        let biases_reshaped = self.biases.clone().insert_axis(ndarray::Axis(0));
        product += &biases_reshaped;
        activate_2d(self.activation_function, &mut product);
        // println!("Output:\n{}", product);
        product
    }

    pub fn backpropagate(&mut self, input: &ArrayView2<f32>, my_output: &ArrayView2<f32>, error: &ArrayView2<f32>) -> Array2<f32> {
       let mut derivative = my_output.to_owned();
        activation_derivative_2d(self.activation_function, &mut derivative);

        let dl_da = *error;
        let grad = &dl_da * &derivative;

        let output = grad.dot(&self.weights.t());
        
        let mut weight_gradients = input.t().dot(&grad);

        //L2 Regularization... it's that easy!
        weight_gradients.scaled_add(self.lambda, &self.weights);

        let mut bias_gradients = grad.sum_axis(ndarray::Axis(0));
        bias_gradients.scaled_add(self.lambda, &self.biases);
        
        const CLIP_THRESHOLD: f32 = 1.0;
        weight_gradients.mapv_inplace(|x| x.clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD));
        bias_gradients.mapv_inplace(|x| x.clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD));
    
        let coefficient = -self.learning_rate / input.nrows() as f32;
        self.biases.scaled_add(coefficient, &bias_gradients);
        self.weights.scaled_add(coefficient, &weight_gradients);
        
        for i in 0..self.biases.len() {
            if self.biases[i].abs() > 200.0 {
                panic!("Biases are big.");
            }
        }
        
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                if self.weights[[i, j]].abs() > 200.0 {
                    panic!("Weights are big.");
                }
            }
        }
        output
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    pub fn get_lambda(&self) -> f32 {
        self.lambda
    }
    pub fn set_weights(&mut self, weights: &ArrayView2<f32>) {
        self.weights = weights.to_owned();
    }

    pub fn set_biases(&mut self, biases: &ArrayView1<f32>) {
        self.biases = biases.to_owned();
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
        for i in 0..self.output_size {
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
            lambda: self.lambda,
            weights: self.weights.clone(),
            biases: self.biases.clone()
        }
    }
}