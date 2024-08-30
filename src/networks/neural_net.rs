use std::default::Default;

use ndarray::{ArrayD, ArrayViewD};

use crate::layers::layer::Layer;

#[derive(Default)]
pub struct NeuralNet {
    layers: Vec<Box<dyn Layer>>,
    num_layers: usize
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

    pub fn forward_pass(&self, input: &ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let mut outputs = vec![];
        outputs.push(input.to_owned());
        //println!("Starting with:\n{}", input);
        let output = self.layers[0].pass(input);
        //println!("Output was: {:?}", output.shape());
        outputs.push(output);
    

        for layer_num in 1..self.num_layers {
            let previous_output = &outputs[layer_num];
            let output = self.layers[layer_num].pass(&previous_output.view());
            //println!("Output was:\n{}", output);
            outputs.push(output);
        }
        outputs.clone()
    }

    pub fn predict(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let outputs = self.forward_pass(input);
        outputs[outputs.len() - 1].clone()
    }

    pub fn backpropagate(&mut self, input: &ArrayViewD<f32>, labels: &ArrayViewD<f32>) -> f32 {
        let outputs = self.forward_pass(input);
        let predictions = &outputs[outputs.len() - 1].view();
        let loss = calculate_bce_loss(predictions, labels);
        let mut error = predictions - labels;
        for layer_num in (0..self.num_layers).rev() {
            let current_input = &outputs[layer_num].view();
            let current_output = &outputs[layer_num+1].view();
            error = self.layers[layer_num].backpropagate(current_input, current_output, &error.view())
        }

        loss
    }
}
    
pub fn calculate_bce_loss(predictions: &ArrayViewD<f32>, labels: &ArrayViewD<f32>) -> f32 {
    let mut loss = 0.0;
    if predictions.shape() != labels.shape() {
        panic!("Mismatched shapes: {:?} predictions and {:?} labels.", predictions.shape(), labels.shape());
    }
    for (index, pred) in predictions.indexed_iter() {
        let epsilon = 1e-7;
        let label = labels[index].clamp(epsilon, 1.0-epsilon);
        loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
    }
    loss / labels.dim()[0] as f32
}