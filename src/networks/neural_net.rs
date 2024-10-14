use std::default::Default;

use ndarray::{ArrayD, ArrayViewD};

use crate::layers::layer::Layer;

#[derive(Default)]
pub struct NeuralNet {
    layers: Vec<Box<dyn Layer>>,
    num_layers: usize,
}

impl NeuralNet {
    pub fn new() -> NeuralNet {
        NeuralNet {
            layers: vec![],
            num_layers: 0,
        }
    }

    ///Add the given layer.
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        if self.num_layers > 0 {
            let input_shape = layer.get_input_shape();
            let previous_output_shape = self.layers[self.num_layers-1].get_output_shape();
            if input_shape != previous_output_shape {
                panic!("Input of current layer and output of previous don't match:\nLayer {} with shape {:?} and layer {} with shape {:?}.",
                        self.num_layers-1, previous_output_shape, self.num_layers, input_shape);
            }
        }
        self.layers.push(layer);
        self.num_layers += 1;
    }

    ///Initializes each layer, currently using Xavier initialization.
    pub fn initialize(&mut self) {
        for layer in &mut self.layers {
            layer.initialize();
        }
    }

    ///Returns a Vec containing the output of each layer.
    pub fn forward_pass(&self, input: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
        let debug = false;
        let mut outputs = vec![];
        outputs.push(input.to_owned());
        if debug {
            println!("Passing layer 0.");
        }
        let output = self.layers[0].pass(input);
        //println!("Output was: {:?}", output.shape());
        outputs.push(output);
    

        for layer_num in 1..self.num_layers {
            if debug {
                println!("Passing layer {layer_num}.");
            }
            let previous_output = &outputs[layer_num];
            let output = self.layers[layer_num].pass(previous_output);
            //println!("Output was:\n{}", output);
            outputs.push(output);
        }
        outputs.clone()
    }

    ///Gets the final output from a given input.
    pub fn predict(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let outputs = self.forward_pass(input);
        outputs[outputs.len() - 1].clone()
    }

    ///Takes a single sample or batch and automatically calculates
    ///and applies the gradients to each layer.
    pub fn backpropagate(&mut self, input: &ArrayD<f32>, labels: &ArrayD<f32>, num_classes: usize) -> f32 {
        let debug = false;
        if debug {
            println!("Forward pass...");
        }
        let outputs = self.forward_pass(input);
        let predictions = &outputs[outputs.len() - 1];
        if predictions.is_any_nan() {
            panic!("NaN(s) in backpropagation.");
        }
        let loss = calculate_bce_loss(predictions, labels, num_classes);
        let mut error = predictions - labels;
        for layer_num in (0..self.num_layers).rev() {
            if debug {
                println!("Backpropagating layer {layer_num}");
            }
            let current_input = &outputs[layer_num];
            let current_output = &outputs[layer_num+1];
            if debug {
                println!("Input: {:?}", current_input.shape());
                println!("Output: {:?}", current_output.shape());
            }
            error = self.layers[layer_num].backpropagate(current_input, current_output, &error)
        }

        loss
    }

    ///Calculates and accumulates the gradients without applying them -- to be used
    ///to average over multiple batches in conjunction with `apply_gradients()`.
    pub fn accumulate_gradients(
        &mut self,
        input: &ArrayD<f32>,
        labels: &ArrayD<f32>,
        num_classes: usize
    ) -> f32 {
        let debug = false;
        if debug {
            println!("Forward pass...");
        }
        let outputs = self.forward_pass(input);
        let predictions = &outputs[outputs.len() - 1];
        if predictions.is_any_nan() {
            panic!("NaN(s) in backpropagation.");
        }
        let loss = calculate_bce_loss(predictions, labels, num_classes);
        let mut error = predictions - labels;
        for layer_num in (0..self.num_layers).rev() {
            if debug {
                println!("Backpropagating layer {layer_num}");
            }
            let current_input = &outputs[layer_num];
            let current_output = &outputs[layer_num+1];
            error = self.layers[layer_num].accumulate_gradients(current_input, current_output, &error)
        }

        loss
    }

    ///Applies the gradients calculated from `accumulate_gradients()`.
    pub fn apply_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.apply_accumulated_gradients();
        }
    }

    ///Sets the gradients of each layer to 0.
    pub fn zero_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.zero_gradients();
        }
    }
}

impl Clone for NeuralNet {
    fn clone(&self) -> Self {
        let mut layers_clone: Vec<Box<dyn Layer>> = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            layers_clone.push(self.layers[i].copy_into_box());
        }
        Self {layers: layers_clone, num_layers: self.num_layers}
    }
}

///Calculates the Binary Cross Entropy between the given prediction and label values, along with
///dividing by the number of classes and batches to get the average loss per class.
///If you're not doing classification, just set `num_classes` to 1.
pub fn calculate_bce_loss(predictions: &ArrayD<f32>, labels: &ArrayD<f32>, num_classes: usize) -> f32 {
    let mut loss = 0.0;
    if predictions.shape() != labels.shape() {
        panic!("Mismatched shapes: {:?} predictions and {:?} labels.", predictions.shape(), labels.shape());
    }
    for (index, prediction) in predictions.indexed_iter() {
        let epsilon = 1e-7;
        let pred = prediction.clamp(epsilon, 1.0-epsilon);
        let label = labels[index].clamp(epsilon, 1.0-epsilon);
        loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
    }
    loss /= num_classes as f32;
    loss / labels.dim()[0] as f32
}

// pub fn calculate_mse_loss(predictions: &ArrayD<f32>, labels: &ArrayD<f32>) -> f32 {
    
// }