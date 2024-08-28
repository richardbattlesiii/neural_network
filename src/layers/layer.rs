use ndarray::{ArrayD, ArrayViewD};

pub trait Layer {
    fn initialize(&mut self);
    fn set_learning_rate(&mut self, rate: f32);
    fn pass(&self, input: &ArrayViewD<f32>) -> ArrayD<f32>;
    fn backpropagate(&mut self, layer_input: &ArrayViewD<f32>,
            layer_output: &ArrayViewD<f32>,
            dl_da: &ArrayViewD<f32>) -> ArrayD<f32>;
    fn get_input_shape(&self) -> Vec<usize>;
    fn get_output_shape(&self) -> Vec<usize>;
}