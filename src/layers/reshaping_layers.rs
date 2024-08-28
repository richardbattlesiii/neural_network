use crate::layers::layer::Layer;
use ndarray::{Array2, ArrayView2, Array4, ArrayView4};

pub struct FlatteningLayer {
    input_shape: (usize, usize, usize),
    size: usize,
}

impl FlatteningLayer {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        FlatteningLayer{
            input_shape,
            size: input_shape.0*input_shape.1*input_shape.2,
        }
    }
}

impl <'a> Layer<'a> for FlatteningLayer {
    type Input = ArrayView4<'a, f32>;
    type Output = Array2<f32>;
    type MyOutputAsAnInput = ArrayView2<'a, f32>;
    type MyInputAsAnOutput = Array4<f32>;
    
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}
    
    fn pass(&self, input: &'a ArrayView4<f32>) -> Array2<f32> {
        input.to_shape((input.dim().0, self.size)).unwrap().to_owned()
    }
    
    fn backpropagate(&mut self, layer_input: &'a ArrayView4<f32>,
            layer_output: &'a ArrayView2<f32>,
            dl_da: &'a ArrayView2<f32>) -> Array4<f32> {
        dl_da.to_shape((dl_da.nrows(), self.input_shape.0, self.input_shape.1, self.input_shape.2)).unwrap().to_owned()
    }
}







pub struct UnflatteningLayer {
    output_shape: (usize, usize, usize),
    size: usize,
}

impl UnflatteningLayer {
    pub fn new(output_shape: (usize, usize, usize)) -> Self {
        UnflatteningLayer{
            output_shape,
            size: output_shape.0*output_shape.1*output_shape.2,
        }
    }
}

impl <'a> Layer<'a> for UnflatteningLayer {
    type Input = ArrayView2<'a, f32>;
    type Output = Array4<f32>;
    type MyOutputAsAnInput = ArrayView4<'a, f32>;
    type MyInputAsAnOutput = Array2<f32>;
    
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}
    
    fn pass(&self, input: &'a ArrayView2<f32>) -> Array4<f32> {
        input.to_shape((input.nrows(), self.output_shape.0, self.output_shape.1, self.output_shape.2)).unwrap().to_owned()
    }
    
    fn backpropagate(&mut self, layer_input: &'a ArrayView2<f32>,
            layer_output: &'a ArrayView4<f32>,
            dl_da: &'a ArrayView4<f32>) -> Array2<f32> {
        dl_da.to_shape((dl_da.dim().0, self.size)).unwrap().to_owned()
    }
}