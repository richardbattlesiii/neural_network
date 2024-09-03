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

        let input = input_dynamic.to_shape((batch_size, self.size*self.channels)).unwrap();
        
        let mut output = Array2::zeros((batch_size, self.size*self.channels));

        for batch in 0..batch_size {
            for i in 0..self.size {
                let mut max = f32::NEG_INFINITY;
                for channel in 0..self.channels {
                    let idx = channel * self.size + i;
                    if input[[batch, idx]] > max {
                        max = input[[batch, idx]];
                    }
                }
        
                let mut sum = 0.0;
                let mut exp_values = vec![0.0; self.channels];
                for channel in 0..self.channels {
                    let idx = channel * self.size + i;
                    let exp_value = (input[[batch, idx]] - max).exp();
                    exp_values[channel] = exp_value;
                    sum += exp_value;
                }
        
                for channel in 0..self.channels {
                    let idx = channel * self.size + i;
                    output[[batch, idx]] = exp_values[channel] / sum;
                }
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
        vec![self.size*self.channels]
    }
    
    fn get_output_shape(&self) -> Vec<usize> {
        vec![self.size*self.channels]
    }
}
