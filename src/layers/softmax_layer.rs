use ndarray::prelude::{Array2, ArrayView2};
use crate::layers::layer::Layer;

pub struct SoftmaxLayer {}

impl SoftmaxLayer {
    pub fn new() -> SoftmaxLayer {
        SoftmaxLayer{}
    }
}
impl<'a> Layer<'a> for SoftmaxLayer {
    type Input = ArrayView2<'a, f32>;
    type Output = Array2<f32>;
    type MyOutputAsAnInput = ArrayView2<'a, f32>;
    type MyInputAsAnOutput = Array2<f32>;
    
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}

    fn pass(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let num_tiles = input.ncols();
        let num_colors = input.ncols() / num_tiles;
        
        //Create an output array with the same shape as the input
        let mut output = Array2::zeros(input.raw_dim());

        for batch in 0..input.nrows() {
            for tile in 0..num_tiles {
                let start = tile * num_colors;
                let end = start + num_colors;

                //Find the max value for numerical stability
                let mut max = f32::NEG_INFINITY;
                for i in start..end {
                    if input[[batch, i]] > max {
                        max = input[[batch, i]];
                    }
                }

                //Compute the exponentials and sum
                let mut sum = 0.0;
                let mut exp_values = vec![0.0; num_colors];
                for i in start..end {
                    let exp_value = (input[[batch, i]] - max).exp();
                    exp_values[i - start] = exp_value;
                    sum += exp_value;
                }

                //Normalize to get softmax values
                for i in start..end {
                    output[[batch, i]] = exp_values[i - start] / sum;
                }
            }
        }

        output
    }

    fn backpropagate(&mut self, layer_input: &'a ArrayView2<'a, f32>,
                layer_output: &'a ArrayView2<'a, f32>,
                dl_da: &'a ArrayView2<'a, f32>) -> Array2<f32> {
        dl_da.to_owned()
    }
}
