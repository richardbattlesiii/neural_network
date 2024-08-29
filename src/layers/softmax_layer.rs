use ndarray::prelude::{ArrayD, ArrayViewD, Array2};
use crate::layers::layer::Layer;

///Currently only supports 2d (batch size x prediction) inputs
pub struct SoftmaxLayer {
    size: usize,
    channels: usize
}

impl SoftmaxLayer {
    pub fn new(size: usize, channels: usize) -> SoftmaxLayer {
        SoftmaxLayer{
            size,
            channels,
        }
    }
}
impl Layer for SoftmaxLayer {
    
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}

    fn pass(&self, input_dynamic: &ArrayViewD<f32>) -> ArrayD<f32> {
        let batch_size = input_dynamic.dim()[0];

        let input = input_dynamic.to_shape((batch_size, self.size)).unwrap();
        
        let mut output = Array2::zeros((batch_size, self.size));

        for batch in 0..batch_size {
            let mut max = f32::NEG_INFINITY;
            for i in 0..self.size {
                if input[[batch, i]] > max {
                    max = input[[batch, i]];
                }
            }

            let mut sum = 0.0;
            let mut exp_values = vec![0.0; self.size];
            for i in 0..self.size {
                let exp_value = (input[[batch, i]] - max).exp();
                exp_values[i] = exp_value;
                sum += exp_value;
            }

            for i in 0..self.size {
                output[[batch, i]] = exp_values[i] / sum;
            }
        }

        output.into_dyn()
    }

    fn backpropagate(&mut self, layer_input: &ArrayViewD<f32>,
                layer_output: &ArrayViewD<f32>,
                dl_da: &ArrayViewD<f32>) -> ArrayD<f32> {
        dl_da.to_owned()
    }
    
    fn get_input_shape(&self) -> Vec<usize> {
        vec![self.size]
    }
    
    fn get_output_shape(&self) -> Vec<usize> {
        vec![self.size]
    }
}
