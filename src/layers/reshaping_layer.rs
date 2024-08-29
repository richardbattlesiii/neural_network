use crate::layers::layer::Layer;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub struct ReshapingLayer {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl ReshapingLayer {
    pub fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        ReshapingLayer {
            input_shape,
            output_shape,
        }
    }
}

impl Layer for ReshapingLayer {
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}
    
    fn pass(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        if input.is_any_nan() {
            println!("Input:\n{}", input);
            panic!("Reshaping layer got NaN input.");
        }
        let batch_size = input.dim()[0];
        let mut output_vec = self.output_shape.clone();
        output_vec.insert(0, batch_size);
        let output_shape = IxDyn(&output_vec);
        input.to_shape(output_shape).unwrap().to_owned()
    }
    
    fn backpropagate(&mut self, layer_input: &ArrayViewD<f32>,
            layer_output: &ArrayViewD<f32>,
            dl_da: &ArrayViewD<f32>) -> ArrayD<f32> {
        let batch_size = dl_da.dim()[0];
        let mut input_vec = self.output_shape.clone();
        input_vec.insert(0, batch_size);
        let input_shape = IxDyn(&input_vec);
        dl_da.to_shape(input_shape).unwrap().to_owned()
    }
    
    fn get_input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
    
    fn get_output_shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}