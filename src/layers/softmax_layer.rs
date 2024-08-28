use ndarray::prelude::{ArrayD, ArrayViewD};
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

    fn pass(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let batch_size = input.dim()[0];
        let num_tiles = input.len()/batch_size/self.channels;
        
        let mut output = ArrayD::zeros(input.raw_dim());

        for batch in 0..batch_size {
            for tile in 0..num_tiles {
                let start = tile * self.channels;
                let end = start + self.channels;

                let mut max = f32::NEG_INFINITY;
                for i in start..end {
                    if input[[batch, i]] > max {
                        max = input[[batch, i]];
                    }
                }

                let mut sum = 0.0;
                let mut exp_values = vec![0.0; self.channels];
                for i in start..end {
                    let exp_value = (input[[batch, i]] - max).exp();
                    exp_values[i - start] = exp_value;
                    sum += exp_value;
                }

                for i in start..end {
                    output[[batch, i]] = exp_values[i - start] / sum;
                }
            }
        }

        output
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
