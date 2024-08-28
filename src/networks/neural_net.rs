use ndarray::{ArrayD, ArrayViewD};

use crate::layers::layer::Layer;

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
        self.layers.push(layer);
        self.num_layers += 1;
    }

    pub fn forward_pass(&self, input: &ArrayViewD<f32>) -> Vec<ArrayD<f32>> {
        let mut outputs = vec![];
        outputs.push(self.layers[0].pass(input));
    
        for layer_num in 0..self.num_layers {
            let converted = outputs[layer_num-1].clone().into_dyn();
            let output = self.layers[layer_num].pass(&converted.view()).clone();
            outputs.push(output.clone());
            drop(output);
        }
        outputs.clone()
    }
}