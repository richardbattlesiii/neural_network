use crate::layers::layer::Layer;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

#[derive(Clone)]
pub struct ReshapingLayer {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl ReshapingLayer {
    pub fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        let mut input_size = 1;
        let mut output_size = 1;
        for i in 0..input_shape.len() {
            input_size *= input_shape[i];
        }
        for i in 0..output_shape.len() {
            output_size *= output_shape[i];
        }
        if input_size != output_size {
            panic!("Mismatched sizes for ReshapingLayer: {input_size} and {output_size}.");
        }
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
    
    fn pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
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
    
    fn backpropagate(&mut self, layer_input: &ArrayD<f32>,
        layer_output: &ArrayD<f32>,
        dl_da: &ArrayD<f32>
    ) -> ArrayD<f32> {
        self.accumulate_gradients(layer_input, layer_output, dl_da)
    }

    fn copy_into_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    
    fn get_input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
    
    fn get_output_shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
    
    fn zero_gradients(&mut self) {}
    
    fn accumulate_gradients(
        &mut self,
        layer_input: &ArrayD<f32>,
        layer_output: &ArrayD<f32>,
        dl_da: &ArrayD<f32>
    ) -> ArrayD<f32> {
        let batch_size = dl_da.dim()[0];
        let mut input_vec = self.input_shape.clone();
        input_vec.insert(0, batch_size);
        let input_shape = IxDyn(&input_vec);
        dl_da.to_shape(input_shape).unwrap().to_owned()
    }
    
    fn apply_accumulated_gradients(&mut self) {}
}