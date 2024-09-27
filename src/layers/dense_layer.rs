use crate::helpers::activation_functions::*;
use num::pow::Pow;
use rand::Rng;
use std::fmt;
use ndarray::{Array1, ArrayView1, Array2, ArrayView2, ArrayD, ArrayViewD};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::layers::layer::Layer;

#[derive(Clone)]
pub struct DenseLayer {
    input_size:usize,
    output_size:usize,

    learning_rate:f32,
    activation_function:u8,
    lambda:f32,

    weights:Array2<f32>,
    biases:Array1<f32>,

    weight_gradients: Array2<f32>,
    bias_gradients: Array1<f32>,

    weight_first_moments: Array2<f32>,
    bias_first_moments: Array1<f32>,
    weight_second_moments: Array2<f32>,
    bias_second_moments: Array1<f32>,
    timestep: i32,

    num_batches: usize,
}

impl Layer for DenseLayer {
    fn initialize(&mut self){
        let xavier = f32::sqrt(6.0/((self.input_size + self.output_size) as f32 ));
        self.weights = Array2::random((self.input_size, self.output_size), Uniform::new(-xavier, xavier));
        self.biases = Array1::random(self.output_size, Uniform::new(-0.01, 0.01));
    }

    fn pass(&self, input_dynamic: &ArrayD<f32>) -> ArrayD<f32> {
        // println!("Input:\n{}", input);
        //println!("Converting shape {:?} to ({} by {:?})", input_dynamic.shape(), input_dynamic.dim()[0], self.input_size);
        let input = input_dynamic.to_shape((input_dynamic.dim()[0], self.input_size)).unwrap();
        let mut product = input.dot(&self.weights).into_dyn();
        let biases_reshaped = self.biases.clone().insert_axis(ndarray::Axis(0));
        product += &biases_reshaped;
        activate(self.activation_function, &mut product);
        // println!("Output:\n{}", product);
        product
    }

    fn backpropagate(
        &mut self,
        input_dynamic: &ArrayD<f32>,
        my_output_dynamic: &ArrayD<f32>,
        error_dynamic: &ArrayD<f32>
    ) -> ArrayD<f32> {
        self.zero_gradients();
        let output = self.accumulate_gradients(input_dynamic, my_output_dynamic, error_dynamic);
        self.apply_accumulated_gradients();
        
        for i in 0..self.biases.len() {
            if self.biases[i].abs() > 100.0 {
                panic!("Biases are big.");
            }
        }
        
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                if self.weights[[i, j]].abs() > 100.0 {
                    panic!("Weights are big.");
                }
            }
        }
        output
    }

    fn copy_into_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }
    
    fn get_input_shape(&self) -> Vec<usize> {
        vec![self.input_size]
    }
    
    fn get_output_shape(&self) -> Vec<usize> {
        vec![self.output_size]
    }
    
    fn zero_gradients(&mut self) {
        self.weight_gradients = Array2::zeros((self.input_size, self.output_size));
        self.bias_gradients = Array1::zeros(self.output_size);
        self.num_batches = 0;
    }

    fn accumulate_gradients(
        &mut self,
        layer_input_dynamic: &ArrayD<f32>,
        layer_output_dynamic: &ArrayD<f32>,
        dl_da_dynamic: &ArrayD<f32>,
    ) -> ArrayD<f32> {
        self.num_batches += 1;

        let batch_size = layer_input_dynamic.dim()[0];
        let input = layer_input_dynamic.to_shape((batch_size, self.input_size)).unwrap();
        let my_output = layer_output_dynamic.to_shape((batch_size, self.output_size)).unwrap();
        let error = dl_da_dynamic.to_shape((batch_size, self.output_size)).unwrap();

        let mut derivative = layer_output_dynamic.to_owned();
        activation_derivative(self.activation_function, &mut derivative);
        let derivative = derivative.to_shape((batch_size, self.output_size)).unwrap();
        let dl_da = error;
        let grad = &dl_da * &derivative;

        let output = grad.dot(&self.weights.t());
        
        let mut weight_gradients = input.t().dot(&grad);
        let mut bias_gradients = grad.sum_axis(ndarray::Axis(0));

        //Gradient clipping
        const CLIP_THRESHOLD: f32 = 2.0;
        weight_gradients.mapv_inplace(|x| x.clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD));
        bias_gradients.mapv_inplace(|x| x.clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD));

        //L2 Regularization... it's that easy!
        weight_gradients.scaled_add(self.lambda, &self.weights);
        bias_gradients.scaled_add(self.lambda, &self.biases);
    
        let coefficient = 1.0 / input.nrows() as f32;

        self.weight_gradients.scaled_add(coefficient, &weight_gradients);
        self.bias_gradients.scaled_add(coefficient, &bias_gradients);

        output.into_dyn()
    }
    
    fn apply_accumulated_gradients(&mut self) {
        self.timestep += 1;
        let coefficient = 1.0/self.num_batches as f32;

        self.weight_gradients *= coefficient;
        self.bias_gradients *= coefficient;

        let beta1 = 0.8;
        let beta2 = 0.9;
        let epsilon = 1e-7;

        let mut weight_first_moments = beta1 * &self.weight_first_moments + (1.0 - beta1) * &self.weight_gradients;
        weight_first_moments = &weight_first_moments / (1.0 - beta1.powi(self.timestep));
        
        let mut bias_first_moments = beta1 * &self.bias_first_moments + (1.0 - beta1) * &self.bias_gradients;
        bias_first_moments = &bias_first_moments / (1.0 - beta1.powi(self.timestep));

        let mut weight_second_moments = beta2 * &self.weight_second_moments + (1.0 - beta2) * &self.weight_gradients * &self.weight_gradients;
        weight_second_moments = &weight_second_moments / (1.0 - beta2.powi(self.timestep));

        let mut bias_second_moments = beta2 * &self.bias_second_moments + (1.0 - beta2) * &self.bias_gradients * &self.bias_gradients;
        bias_second_moments = &bias_second_moments / (1.0 - beta2.powi(self.timestep));

        self.weight_first_moments = weight_first_moments.clone();
        self.bias_first_moments = bias_first_moments.clone();
        self.weight_second_moments = weight_second_moments.clone();
        self.bias_second_moments = bias_second_moments.clone();

        let weight_change = weight_first_moments / (weight_second_moments.sqrt() + epsilon);
        let bias_change = bias_first_moments / (bias_second_moments.sqrt() + epsilon);
        
        self.weights.scaled_add(-self.learning_rate, &weight_change);
        self.biases.scaled_add(-self.learning_rate, &bias_change);
        self.zero_gradients();
    }
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        learning_rate: f32,
        lambda: f32,
        activation_function: u8,
    ) -> DenseLayer {
        DenseLayer {
            input_size,
            output_size,

            learning_rate,
            activation_function,
            lambda,

            weights: Array2::zeros((input_size, output_size)),
            biases: Array1::zeros(output_size),

            weight_gradients: Array2::zeros((input_size, output_size)),
            bias_gradients: Array1::zeros(output_size),

            weight_first_moments: Array2::zeros((input_size, output_size)),
            bias_first_moments: Array1::zeros(output_size),
            weight_second_moments: Array2::zeros((input_size, output_size)),
            bias_second_moments: Array1::zeros(output_size),
            timestep: 0,

            num_batches: 0,
        }
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