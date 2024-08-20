use crate::*;
use layers::dense_layer::DenseLayer;
use layers::layer::*;

#[derive(Default)]
#[derive(Clone)]
pub struct DenseNet {
    layers: Vec<DenseLayer>,
    num_layers: usize,
}

impl DenseNet {
    pub fn new(layer_sizes: &Vec<usize>, activation_functions: &Vec<u8>) -> DenseNet {
        let mut layers = vec![];
        let num_layers = layer_sizes.len()-1;
        for layer in 0..layer_sizes.len()-1 {
            layers.push(DenseLayer::new(
                layer_sizes[layer],
                layer_sizes[layer+1],
                0.1,
                activation_functions[layer],
            ));
        }

        DenseNet {
            layers,
            num_layers,
        }
    }

    pub fn new_with_arrays(layer_sizes: &[usize; NUMBER_OF_LAYERS], activation_functions: &[u8; NUMBER_OF_LAYERS-1]) -> DenseNet {
        let mut layers = vec![];
        let num_layers = layer_sizes.len()-1;
        for layer in 0..layer_sizes.len()-1 {
            layers.push(DenseLayer::new(
                layer_sizes[layer],
                layer_sizes[layer+1],
                0.1,
                activation_functions[layer],
            ));
        }

        DenseNet {
            layers,
            num_layers,
        }
    }
    
    pub fn set_learning_rate(&mut self, rate: f32) {
        for layer in 0..self.num_layers {
            self.layers[layer].set_learning_rate(rate);
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.layers[0].learning_rate
    }

    pub fn initialize(&mut self) {
        for layer in 0..self.num_layers {
            self.layers[layer].initialize();
        }
    }

    pub fn set_parameters_manually(&mut self, weights: &Vec<Matrix>, biases: &Vec<Matrix>) {
        for layer in 0..self.num_layers {
            self.layers[layer].set_weights(&weights[layer]);
            self.layers[layer].set_biases(&biases[layer]);
        }
    }

    pub fn get_parameters(&self) -> Vec<(&Matrix, &Matrix)> {
        let mut output = vec![];
        for layer in 0..self.num_layers {
            output.push(self.layers[layer].get_parameters());
        }

        output
    }

    pub fn forward_pass(&self, input: Matrix) -> Vec<Matrix> {
        let mut all_outputs = vec![];
        all_outputs.push(input.copy());
        for layer in 0..self.layers.len() {
            let passed = self.layers[layer].pass(InputTypeEnum::Single(&all_outputs[layer]));
            match passed {
                OutputTypeEnum::Single(pasta) => {
                    all_outputs.push(pasta);
                },
                OutputTypeEnum::Batch(_) => panic!("Got a batch output from a single pass of a dense layer."),
            }
        }
    
        all_outputs
    }
    
    pub fn batch_pass(&self, input: &Matrix) -> Vec<Matrix> {
        let mut all_outputs = vec![];
        all_outputs.push(input.copy());
        for layer in 0..self.layers.len() {
            let passed = self.layers[layer].pass(InputTypeEnum::Single(&all_outputs[layer]));
            match passed {
                OutputTypeEnum::Single(output) => {
                    all_outputs.push(output);
                },
                OutputTypeEnum::Batch(_) => panic!("Got a batch output from a batch pass of a dense layer, but like, wrongly. Idk, you figure it out."),
            }
        }
    
        all_outputs
    }
    
    pub fn predict(&self, input: &Matrix) -> Matrix {
        let mut current_input = input.copy();
        for layer in 0..self.layers.len() {
            let passed = self.layers[layer].pass(InputTypeEnum::Single(&current_input));
            match passed {
                OutputTypeEnum::Single(output) => {
                    current_input = output
                },
                OutputTypeEnum::Batch(_) => panic!("Got a batch output from a single pass of a dense layer."),
            }
        }
        current_input
    }

    pub fn back_prop(&mut self, inputs: &Matrix, labels: &Matrix) {
        for i in 0..inputs.rows {
            let input = inputs.sub_matrix(i, 0, 1, inputs.cols);
            let label = labels.sub_matrix(i, 0, 1, inputs.cols);
    
            let all_outputs = self.forward_pass(input);
    
            // Calculate initial error with output layer derivative
            let mut current_error = all_outputs[all_outputs.len() - 1].copy();
            current_error.subtract(&label); //Derivative of loss function.
            
            for layer in (0..self.layers.len()).rev() {
                let current_layer = &mut self.layers[layer];
                // Propagate error through activation derivative
                current_error = current_layer.backpropagate(&all_outputs[layer], &all_outputs[layer + 1], &current_error);
            }
        }
    }
    
    pub fn batch_backpropagate(&mut self, inputs: &Matrix, labels: &Matrix) -> f32 {
        let all_outputs = self.batch_pass(inputs);
    
        // Calculate initial error with output layer derivative
        let mut current_errors = all_outputs[all_outputs.len() - 1].copy();
        let output = DenseNet::calculate_mse_loss(&current_errors, labels);
        // current_errors = self.custom_loss_derivative(&current_errors, &labels);
        current_errors.subtract(&labels); //Derivative of loss function.
        for layer in (0..self.layers.len()).rev() {
            let current_layer = &mut self.layers[layer];
            // Propagate error through activation derivative
            let passed_error = current_layer.batch_backpropagate(InputTypeEnum::Single(&all_outputs[layer]),
                    OutputTypeEnum::Single(&all_outputs[layer + 1]),
                    OutputTypeEnum::Single(&current_errors));
            match passed_error {
                InputTypeEnum::Single(current_error) => current_errors=current_error,
                InputTypeEnum::Batch(_) => todo!(),
            }
        }
        output
    }

    pub fn add_noise(&mut self, range: f32) {
        for layer in 0..self.num_layers {
            self.layers[layer].add_noise(range);
        }
    }
    
    pub fn calculate_bce_loss(predictions: &Matrix, labels: &Matrix) -> f32 {
        let mut loss = 0.0;
        for i in 0..labels.rows {
            for j in 0..labels.cols {
                let pred = predictions.values[i*labels.cols + j];
                let label = labels.values[i*labels.cols + j];
                loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
            }
        }
        loss / labels.rows as f32
    }
    
    pub fn calculate_mse_loss(predictions: &Matrix, labels: &Matrix) -> f32 {
        let mut loss = 0.0;
        for i in 0..labels.rows {
            for j in 0..labels.cols {
                let pred = predictions.values[i*labels.cols + j];
                let label = labels.values[i*labels.cols + j];
                loss += (label-pred)*(label-pred);
            }
        }
        loss / labels.rows as f32
    }
    
    pub fn custom_loss_derivative(&self, predictions: &Matrix, labels: &Matrix) -> Matrix {
        let rows = predictions.rows;
        let cols = predictions.cols;
        let mut output_values = Vec::with_capacity(rows*cols);
        for row in 0..rows {
            for col in 0..cols {
                let prediction = predictions.values[row*cols+col];
                let label = labels.values[row*cols+col];
                if (label == 0.0) ^ (f32::round(prediction) != 0.0) {
                    output_values.push(NEGATIVE_ONE_WRONG_PENALTY*(prediction-label));
                }
                else {
                    output_values.push(prediction-label);
                }
            }
        }
    
        Matrix {
            values: output_values,
            rows,
            cols
        }
    }
    
    pub fn rose_decay(&mut self, epoch: u32, low: f32, high: f32, oscillate_forever: bool,
            oscillation_coefficient: f32, oscillation_parameter: f32, exponential_parameter: f32) -> f32 {
        let rose_decayed_learning_rate;
        if !oscillate_forever {
            let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter);
            rose_decayed_learning_rate = oscillation_coefficient * exponential_decay*f32::sin(oscillation_parameter * epoch as f32)
            + exponential_decay + low;
        }
        else {
            let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter) + low;
            rose_decayed_learning_rate = exponential_decay * (oscillation_coefficient * f32::sin(oscillation_parameter * epoch as f32) + low)
            + exponential_decay;
        }
        self.set_learning_rate(rose_decayed_learning_rate);
        rose_decayed_learning_rate
    }
}