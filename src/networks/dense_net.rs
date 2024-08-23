use crate::*;
use layers::dense_layer::DenseLayer;
use ndarray::{Array1, Array2, s};
#[derive(Default)]
#[derive(Clone)]
/*
    Pretty self-explanatory. A dense neural network, so it only has dense layers.
    Eventually I might add stuff from my Java project like output layers for softmax
    and flattening / unflattening layers. But those aren't really needed for a dense net.
    (okay maybe the output layer would be good... adding it to the list)
*/
pub struct DenseNet {
    layers: Vec<DenseLayer>,
    num_layers: usize,
}

impl DenseNet {
    ///Make a new DenseNet in a concise way just using two vectors.
    ///Default learning rate must be changed with set_learning_rate() or rose_decay().
    ///Defeault lambda for L2 regularization must be changed with 
    pub fn new_with_vectors(layer_sizes: &[usize], activation_functions: &[u8]) -> DenseNet {
        let mut layers = vec![];
        //Note the - 1
        let num_layers = layer_sizes.len()-1;
        for layer in 0..layer_sizes.len()-1 {
            layers.push(DenseLayer::new(
                //Input size
                layer_sizes[layer],
                //Output size
                layer_sizes[layer+1],
                //Learning rate
                0.1,
                //Lambda
                0.01,
                //Activation function
                activation_functions[layer],
            ));
        }

        DenseNet {
            layers,
            num_layers,
        }
    }

    ///Like new_with_vectors, but with arrays instead. Useful because multithreading is easier if
    ///layer_sizes and activation_functions are consts.
    pub fn new_with_arrays(layer_sizes: &[usize; NUMBER_OF_LAYERS], activation_functions: &[u8; NUMBER_OF_LAYERS-1]) -> DenseNet {
        let mut layers = vec![];
        let num_layers = layer_sizes.len()-1;
        for layer in 0..num_layers {
            layers.push(DenseLayer::new(
                layer_sizes[layer],
                layer_sizes[layer+1],
                0.1,
                0.01,
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
        self.layers[0].get_learning_rate()
    }

    pub fn set_lambda(&mut self, lambda: f32) {
        for layer in 0..self.num_layers {
            self.layers[layer].set_lambda(lambda);
        }
    }

    pub fn initialize(&mut self) {
        for layer in 0..self.num_layers {
            self.layers[layer].initialize();
        }
    }

    //Will use eventually in combination with reading parameters from a text file
    pub fn set_parameters_manually(&mut self, weights: &[Array2<f32>], biases: &[Array1<f32>]) {
        for layer in 0..self.num_layers {
            self.layers[layer].set_weights(&weights[layer]);
            self.layers[layer].set_biases(&biases[layer]);
        }
    }

    pub fn get_parameters(&self) -> Vec<(&Array2<f32>, &Array1<f32>)> {
        let mut output = vec![];
        for layer in 0..self.num_layers {
            output.push(self.layers[layer].get_parameters());
        }

        output
    }

    //Returns the output from each layer.
    pub fn forward_pass(&self, input: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut all_outputs = vec![];
        all_outputs.push(input.clone());
        for layer in 0..self.num_layers {
            //Pass the previous output through the current layer
            let passed = self.layers[layer].pass(&all_outputs[layer]);
            all_outputs.push(passed);
        }
    
        all_outputs
    }
    
    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        let outputs = self.forward_pass(input);
        outputs[outputs.len() - 1].clone()
    }

    pub fn backpropagate(&mut self, input: &Array2<f32>, label: &Array2<f32>) -> f32 {
        let all_outputs = self.forward_pass(input);

        //Calculate initial error
        let mut current_error = all_outputs[all_outputs.len() - 1].clone();
        let output = DenseNet::calculate_bce_loss(&all_outputs[all_outputs.len()-1], label);
        //Derivative of MSE.
        current_error -= label;
        for layer in (0..self.num_layers).rev() {
            let current_layer = &mut self.layers[layer];
            // Propagate error through activation derivative
            current_error = current_layer.backpropagate(&all_outputs[layer], &all_outputs[layer + 1], &current_error);
        }

        output
    }
    
    pub fn add_noise(&mut self, range: f32) {
        for layer in 0..self.num_layers {
            self.layers[layer].add_noise(range);
        }
    }
    
    pub fn calculate_bce_loss(predictions: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let mut loss = 0.0;
        for i in 0..labels.nrows() {
            for j in 0..labels.ncols() {
                let pred = predictions[[i, j]];
                let label = labels[[i, j]];
                loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
            }
        }
        loss / labels.nrows() as f32
    }
    
    pub fn calculate_mse_loss(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let mut loss = 0.0;
        for i in 0..labels.nrows() {
            for j in 0..labels.ncols() {
                let pred = predictions[[i, j]];
                let label = labels[[i, j]];
                let diff = label - pred;
                loss += (diff)*(diff);
            }
        }
        loss /= labels.len() as f32;

        let mut l2_penalty = 0.0;
        for layer in 0..self.num_layers {
            let weights = self.layers[layer].get_parameters().0;
            for row in 0..weights.nrows() {
                for col in 0..weights.ncols() {
                    let weight = weights[[row, col]];
                    l2_penalty += weight*weight * self.layers[layer].get_lambda() / 2.0;
                }
            }
        }
        loss + l2_penalty
    }
    
    // pub fn custom_loss_derivative(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Array2<f32> {
    //     let mut output = Array2::zeros((predictions.nrows(), predictions.ncols()));
    //     for row in 0..predictions.nrows() {
    //         for col in 0..predictions.ncols() {
    //             let prediction = predictions[[row, col]];
    //             let label = labels[[row, col]];
    //             if (label == 0.0) ^ (f32::round(prediction) != 0.0) {
    //                 output[[row, col]] = NEGATIVE_ONE_WRONG_PENALTY*(prediction-label);
    //             }
    //             else {
    //                 output[[row, col]] = prediction-label;
    //             }
    //         }
    //     }
    
    //     output
    // }
    
    pub fn rose_decay_learning_rate(&mut self, epoch: u32, low: f32, high: f32, oscillate_forever: bool,
            oscillation_coefficient: f32, oscillation_parameter: f32, exponential_parameter: f32) -> f32 {
        let rose_decayed_learning_rate =
            if oscillate_forever {
                let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter) + low;
                exponential_decay * (oscillation_coefficient * f32::sin(oscillation_parameter * epoch as f32) + low)
                + exponential_decay
            }
            else {
                let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter);
                oscillation_coefficient * exponential_decay*f32::sin(oscillation_parameter * epoch as f32)
                + exponential_decay + low
            };
        self.set_learning_rate(rose_decayed_learning_rate);
        rose_decayed_learning_rate
    }

    pub fn exponentially_decay_learning_rate(&mut self, epoch: u32, parameter_one: f32, parameter_two: f32,parameter_three: f32) -> f32 {
        let result = parameter_one*f32::exp(epoch as f32 * parameter_two) + parameter_three;
        self.set_learning_rate(result);
        result
    }

    pub fn oscillate_learning_rate(&mut self, epoch: u32, parameter_one: f32, parameter_two: f32,parameter_three: f32) -> f32 {
        let result = parameter_one*f32::exp(epoch as f32 * parameter_two) + parameter_three;
        self.set_learning_rate(result);
        result
    }

    pub fn write_net_params_to_string(&self) -> String {
        let mut output = "".to_string();
        for layer in 0..self.num_layers {
            output += &(self.layers[layer].write_params_to_string()+"\n");
        }

        output
    }
}