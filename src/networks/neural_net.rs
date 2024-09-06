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

    pub fn initialize(&mut self) {
        for layer in &mut self.layers {
            layer.initialize();
        }
    }

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
            let output = self.layers[layer_num].pass(&previous_output);
            //println!("Output was:\n{}", output);
            outputs.push(output);
        }
        outputs.clone()
    }

    pub fn predict(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let outputs = self.forward_pass(input);
        outputs[outputs.len() - 1].clone()
    }

    pub fn backpropagate(&mut self, input: &ArrayD<f32>, labels: &ArrayD<f32>, num_classes: usize) -> f32 {
        let debug = false;
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
            error = self.layers[layer_num].backpropagate(current_input, current_output, &error)
        }

        loss
    }
}
    
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