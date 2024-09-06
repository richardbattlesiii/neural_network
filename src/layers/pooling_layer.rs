use super::layer::Layer;
use ndarray::{ArrayD, ArrayViewD, Array2, s};

pub const POOLING_MAX: u8 = 0;
pub const POOLING_AVERAGE: u8 = 1;

///Only accepts 2d I/O for now.
pub struct PoolingLayer {
    input_dimensions: Vec<usize>,
    output_dimensions: Vec<usize>,
    pooling_dimensions: Vec<usize>,
    pooling_mode: u8,
}

impl PoolingLayer {
    pub fn new(
            input_dimensions: Vec<usize>,
            pooling_dimensions: Vec<usize>,
            pooling_mode: u8) -> PoolingLayer {
        let mut output_dimensions: Vec<usize> = Vec::with_capacity(input_dimensions.len());
        for i in 0.. input_dimensions.len() {
            output_dimensions[i] = input_dimensions[i] / pooling_dimensions[i];
            //Check that the pooling dimensions are valid
            if pooling_dimensions[i] * output_dimensions[i] != input_dimensions[i] {
                panic!("Invalid pooling dimensions: {} -> {} by {}", input_dimensions[i], output_dimensions[i], pooling_dimensions[i]);
            }
        }
        PoolingLayer {
            input_dimensions,
            output_dimensions,
            pooling_dimensions,
            pooling_mode,
        }
    }
}

impl Layer for PoolingLayer {
    ///Does nothing.
    fn initialize(&mut self) {}
    ///Does nothing.
    fn set_learning_rate(&mut self, rate: f32) {}

    fn pass(&self, input_dynamic: &ArrayD<f32>) -> ArrayD<f32> {
        let input_rows = self.input_dimensions[0];
        let input_cols = self.input_dimensions[1];

        let pool_rows = self.pooling_dimensions[0];
        let pool_cols = self.pooling_dimensions[1];

        let output_rows = input_rows / pool_rows;
        let output_cols = input_cols / pool_rows;

        let input = input_dynamic.to_shape((input_rows, input_cols)).unwrap();
        let mut output: Array2<f32> = Array2::zeros((output_rows, output_cols));

        for row in 0..output_rows {
            for col in 0..output_cols {
                let row_start = row * pool_rows;
                let row_end = row_start + pool_rows;
                let col_start = col * pool_cols;
                let col_end = col_start + pool_cols;

                let slice = input.slice(s![row_start..row_end, col_start..col_end]);

                match self.pooling_mode {
                    POOLING_MAX => {
                        let max = slice
                                .iter()
                                .copied()
                                .max_by(|a, b| a.partial_cmp(b).unwrap())
                                .unwrap();
                        output[[row, col]] = max;
                    },
                    POOLING_AVERAGE => {
                        let average = slice.sum() / slice.len() as f32;
                        output[[row, col]] = average;
                    }
                    _ => panic!("Invalid pooling mode."),
                }
            }
        }
        
        output.into_dyn()
    }

    fn backpropagate(&mut self, layer_input_dynamic: &ArrayD<f32>,
            layer_output_dynamic: &ArrayD<f32>,
            dl_da_dynamic: &ArrayD<f32>) -> ArrayD<f32> {
        let input_rows = self.input_dimensions[0];
        let input_cols = self.input_dimensions[1];

        let pool_rows = self.pooling_dimensions[0];
        let pool_cols = self.pooling_dimensions[1];

        let output_rows = input_rows / pool_rows;
        let output_cols = input_cols / pool_rows;

        let mut dl_dx: Array2<f32> = Array2::zeros((input_rows, input_cols));
        let dl_da = dl_da_dynamic.to_shape((output_rows, output_cols)).unwrap();

        match self.pooling_mode {
            POOLING_MAX => {
                let layer_input = layer_input_dynamic.to_shape((input_rows, input_cols)).unwrap();
                let layer_output = layer_output_dynamic.to_shape((output_rows, output_cols)).unwrap();
                for row in 0..input_rows {
                    for col in 0..input_cols {
                        let output_row = row / pool_rows;
                        let output_col = col / pool_cols;
                        if layer_input[[row, col]] == layer_output[[output_row, output_col]] {
                            dl_dx[[row, col]] = dl_da[[output_row, output_col]];
                        }
                    }
                }
                return dl_dx.into_dyn();
            },
            POOLING_AVERAGE => {
                let mut pooling_size = 1;
                for i in 0..self.pooling_dimensions.len() {
                    pooling_size *= self.pooling_dimensions[i];
                }
                for row in 0..input_rows {
                    for col in 0..input_cols {
                        let output_row = row / pool_rows;
                        let output_col = col / pool_cols;
                        dl_dx[[row, col]] = dl_da[[output_row, output_col]] / pooling_size as f32;
                    }
                }
                return dl_dx.into_dyn();
            },
            _ => panic!("Invalid pooling mode in backpropagate... how did it not get an error in the forward pass???"),
        }
    }

    fn get_input_shape(&self) -> Vec<usize> {
        self.input_dimensions.clone()
    }

    fn get_output_shape(&self) -> Vec<usize> {
        let mut output = self.input_dimensions.clone();
        for i in 0..output.len() {
            output[i] /= self.pooling_dimensions[i];
        }
        output
    }
}