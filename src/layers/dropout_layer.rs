use super::layer::Layer;
use ndarray::{ArrayD, ArrayViewD};
use ndarray_rand::{rand, rand::Rng};

pub const DROPOUT_MULTIPLY: u8 = 0;
pub const DROPOUT_ADD: u8 = 1;
pub const DROPOUT_ZERO: u8 = 2;

pub struct DropoutLayer {
    dimensions: Vec<usize>,
    dropout_chance: f32,
    dropout_magnitude: f32,
    dropout_mode: u8,
}

impl DropoutLayer {
    pub fn new(dimensions: Vec<usize>, dropout_chance: f32, dropout_magnitude: f32, dropout_mode: u8) -> Self {
        DropoutLayer {
            dimensions,
            dropout_chance,
            dropout_magnitude,
            dropout_mode
        }
    }

    pub fn new_zero(dimensions: Vec<usize>, dropout_chance: f32) -> DropoutLayer {
        DropoutLayer {
            dimensions,
            dropout_chance,
            dropout_magnitude: 0.,
            dropout_mode: DROPOUT_ZERO,
        }
    }
}

impl Layer for DropoutLayer {
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}

    fn pass(&self, input: &ndarray::ArrayViewD<f32>) -> ndarray::ArrayD<f32> {
        let mut output: ArrayD<f32> = ArrayD::zeros(input.raw_dim());
        let mut rng = rand::thread_rng();
        for (index, value) in input.indexed_iter() {
            let chance = rng.gen::<f32>();
            if chance < self.dropout_chance {
                match self.dropout_mode {
                    DROPOUT_ADD => {
                        let random = rng.gen::<f32>();
                        output[&index] = input[&index] + self.dropout_magnitude*(random-0.5)
                    },
                    DROPOUT_MULTIPLY => {
                        let random = rng.gen::<f32>();
                        output[&index] = input[&index] *
                                (2. * self.dropout_magnitude * random + 1. - self.dropout_magnitude)
                    },
                    DROPOUT_ZERO => {
                        output[&index] = 0.;
                    }
                    _ => panic!("Invalid dropout mode."),
                }
            }
            else {
                output[&index] = input[&index];
            }
        }
        output
    }

    fn backpropagate(&mut self, layer_input: &ndarray::ArrayViewD<f32>,
            layer_output: &ndarray::ArrayViewD<f32>,
            dl_da: &ndarray::ArrayViewD<f32>) -> ndarray::ArrayD<f32> {
        dl_da.to_owned()
    }

    fn get_input_shape(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    fn get_output_shape(&self) -> Vec<usize> {
        self.dimensions.clone()
    }
}