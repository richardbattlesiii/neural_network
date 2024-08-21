use std::any::TypeId;

use crate::helpers::matrix::Matrix;
use crate::helpers::activation_functions::*;
use crate::layers::layer::{Layer, InputTypeEnum, OutputTypeEnum};
use rand::Rng;
use std::fmt;

pub struct DenseLayer {
    input_size:usize,
    output_size:usize,

    pub learning_rate:f32,
    pub activation_function:u8,

    weights:Matrix,
    biases:Matrix
}

impl Layer<Matrix, Matrix> for DenseLayer {
    fn initialize(&mut self){
        let mut rng = rand::thread_rng();
        let x = f32::sqrt(6.0/((self.input_size + self.output_size) as f32 ));
        for row in 0..self.input_size {
            for col in 0..self.output_size {
                self.weights.set(row, col, x * (rng.gen::<f32>()*2.0-1.0));
                //self.weights.values[row][col] = 0.01*(col as f32)*(col as f32)*(row as f32);
            }
            self.weights.set(0, row, rng.gen::<f32>()*2.0-1.0);
        }
    }

    fn pass(&self, inputs_enum: InputTypeEnum<&Matrix>) -> OutputTypeEnum<Matrix> {
        match inputs_enum {
            InputTypeEnum::Single(input) => {
                if input.rows == 1 {
                    let mut product = input.multiply(&self.weights);
                    product.add(&self.biases);
                    activate(self.activation_function, &mut product);
                    OutputTypeEnum::Single(product)
                }
                else {
                    let repeated_biases = self.biases.repeat(input.rows, true);
                    let mut product = input.multiply(&self.weights);
                    product.add(&repeated_biases);
                    activate(self.activation_function, &mut product);
                    OutputTypeEnum::Single(product)
                }
            }
            InputTypeEnum::Batch(_) => {
                panic!("Gave a Vec<&Matrix> to a dense layer.");
            }
        }
    }

    fn backpropagate(&mut self, input: &Matrix, my_output: &Matrix, error: &Matrix) -> Matrix {
        let mut dl_da = activation_derivative(self.activation_function, my_output);
        dl_da.element_multiply(error);

        let weight_gradients = input.transpose().multiply(&dl_da);
        let bias_gradients = error;
        let output = error.multiply(&self.weights);
        for col in 0..self.output_size {
            self.biases.values[col] += self.learning_rate*bias_gradients.values[col];
        }
        for row in 0..self.input_size {

            for col in 0..self.output_size {
                self.weights.values[row*self.weights.cols+col] += self.learning_rate*weight_gradients.values[row*weight_gradients.cols+col];
            }
        }

        output
    }

    fn batch_backpropagate(&mut self, inputs_enum: InputTypeEnum<&Matrix>, my_outputs_enum: OutputTypeEnum<&Matrix>, errors_enum: OutputTypeEnum<&Matrix>) -> InputTypeEnum<Matrix> {
        match inputs_enum {
            InputTypeEnum::Single(inputs) => {
                match my_outputs_enum {
                    OutputTypeEnum::Single(my_outputs) => {
                        match errors_enum {
                            OutputTypeEnum::Single(errors) => {
                                let mut error_copy = errors.copy();
                                let grad = error_copy.element_multiply(&activation_derivative(self.activation_function, my_outputs));
                                
                                let mut weight_gradients = grad.transpose().multiply(inputs).transpose();
                                weight_gradients.multiply_scalar(1.0/(inputs.rows as f32));
                        
                                let output = self.weights.multiply(&grad.transpose()).transpose();
                                for col in 0..self.output_size {
                                    let mut average = 0.0;
                                    for sample in 0..inputs.rows {
                                        average += grad.values[sample*grad.cols+col];
                                    }
                                    average /= inputs.rows as f32;
                                    self.biases.values[col] -= self.learning_rate * average;
                                }
                                for row in 0..self.input_size {
                                    for col in 0..self.output_size {
                                        //println!("Weights: {}, Gradients: {}, row: {}, col: {}", self.weights.get_shape(), weight_gradients.get_shape(), row, col);
                                        self.weights.values[row*self.weights.cols+col] -= self.learning_rate*weight_gradients.values[row*weight_gradients.cols+col]/inputs.rows as f32;
                                    }
                                }
                        
                                InputTypeEnum::Single(output)
                            }
                            OutputTypeEnum::Batch(_) => {
                                panic!();
                            }
                        }
                    }
                    OutputTypeEnum::Batch(_) => {
                        panic!();
                    }
                }
            }
            InputTypeEnum::Batch(_) => {
                panic!();
            }
        }
        
    }
    
    fn get_input_type_id(&self) -> TypeId {
        TypeId::of::<Matrix>()
    }
    
    fn get_output_type_id(&self) -> TypeId {
        TypeId::of::<Matrix>()
    }
}

impl fmt::Display for DenseLayer{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Weights:\n{}Biases:\n{}", self.weights, self.biases)?;
        Ok(())
    }
}
impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, learning_rate: f32, activation_function: u8) -> DenseLayer {
        DenseLayer {
            input_size,
            output_size,
            learning_rate,
            activation_function,
            weights: Matrix {
                values:vec![0.0; input_size*output_size],
                rows: input_size,
                cols: output_size
            },
            biases: Matrix {
                values:vec![0.0; output_size],
                rows: 1,
                cols: output_size
            }
        }
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }

    pub fn set_weights(&mut self, weights: &Matrix) {
        self.weights = weights.copy();
    }

    pub fn set_biases(&mut self, biases: &Matrix) {
        self.biases = biases.copy();
    }

    pub fn get_parameters(&self) -> (&Matrix, &Matrix) {
        (&self.weights, &self.biases)
    }

    pub fn add_noise(&mut self, range: f32) {
        let mut rng = rand::thread_rng();
        for node in 0..self.output_size {
            for input_node in 0..self.input_size {
                self.weights.values[node*self.input_size+input_node] *= (rng.gen::<f32>() - 0.5) * range + 1.0;
            }
            self.biases.values[node] *= (rng.gen::<f32>() - 0.5) * range + 1.0;
        }
    }

    pub fn write_params_to_string(&self) -> String {
        let mut output = "".to_string();
        for i in 0..self.weights.values.len() {
            output += &(self.weights.values[i].to_string()+" ");
        }
        output += "\n";
        for i in 0..self.biases.values.len() {
            output += &(self.biases.values[i].to_string()+" ");
        }
        output
    }
}

impl Clone for DenseLayer {
    fn clone(&self) -> DenseLayer {
        DenseLayer {
            input_size: self.input_size,
            output_size: self.output_size,
            learning_rate: self.learning_rate,
            activation_function: self.activation_function,
            weights: self.weights.copy(),
            biases: self.biases.copy()
        }
    }
}