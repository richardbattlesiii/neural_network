use ndarray::ArrayD;
use crate::{activate, activation_derivative, helpers::activation_functions, layers::layer::Layer};

#[derive(Clone)]
pub struct ActivationLayer {
    shape: Vec<usize>,
    activation_function: u8,
}

impl Layer for ActivationLayer {

    ///Does nothing.
    fn initialize(&mut self) {}

    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}

    fn pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let mut output = input.to_owned();
        activate(self.activation_function, &mut output);
        output
    }

    fn backpropagate(
        &mut self,
        layer_input: &ArrayD<f32>,
        layer_output: &ArrayD<f32>,
        dl_da: &ArrayD<f32>
    ) -> ArrayD<f32> {
        self.accumulate_gradients(layer_input, layer_output, dl_da)
    }

    fn get_input_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn get_output_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn copy_into_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    ///Does nothing.
    fn zero_gradients(&mut self) {}

    fn accumulate_gradients(
        &mut self,
        layer_input: &ArrayD<f32>,
        layer_output: &ArrayD<f32>,
        dl_da: &ArrayD<f32>
    ) -> ArrayD<f32> {
        let mut derivative = layer_input.to_owned();
        activation_derivative(self.activation_function, &mut derivative);
        dl_da * derivative
    }

    fn apply_accumulated_gradients(&mut self) {}
}

impl ActivationLayer {
    pub fn new(shape: &Vec<usize>, activation_function: u8) -> ActivationLayer {
        ActivationLayer {
            shape: shape.to_owned(),
            activation_function
        }
    }
}