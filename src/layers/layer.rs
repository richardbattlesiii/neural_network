use ndarray::{ArrayD, ArrayViewD};

pub trait Layer {
    fn initialize(&mut self);
    fn set_learning_rate(&mut self, rate: f32);
    fn pass(&self, input: &ArrayD<f32>) -> ArrayD<f32>;
    fn backpropagate(&mut self, layer_input: &ArrayD<f32>,
            layer_output: &ArrayD<f32>,
            dl_da: &ArrayD<f32>) -> ArrayD<f32>;
    fn get_input_shape(&self) -> Vec<usize>;
    fn get_output_shape(&self) -> Vec<usize>;
}