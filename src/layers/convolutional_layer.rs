use crate::helpers::matrix::*;
use crate::helpers::activation_functions::*;
use rand::random;
use rand::Rng;
use std::fmt;
use std::thread::current;
use crate::matrix;
use crate::matrix::*;
use super::layer::{Layer, InputTypeEnum, OutputTypeEnum};

pub struct ConvolutionalLayer {
    input_size:usize,
    output_size:usize,
    input_channels:usize,

    pub learning_rate:f32,
    pub activation_function:u8,

    filters:Vec<Vec<Matrix>>,
    biases:Matrix,
    num_filters: usize,
    filter_size: usize,

    stride: usize,
    padding: usize,
}

impl ConvolutionalLayer {
    pub fn new(
            input_size: usize,
            output_size: usize,
            input_channels: usize,
            learning_rate: f32,
            activation_function: u8,
            num_filters: usize,
            filter_size: usize,
            stride: usize,
            padding: usize) -> ConvolutionalLayer {

        ConvolutionalLayer {
            input_size,
            output_size,
            input_channels,
            learning_rate,
            activation_function,
            filters: Vec::with_capacity(num_filters),
            biases: Matrix {
                values:vec![0.0; num_filters],
                rows: 1,
                cols: num_filters
            },
            num_filters,
            filter_size,
            stride,
            padding,
        }
    }
}

impl Layer<Vec<Matrix>, Vec<Matrix>> for ConvolutionalLayer {
    

    fn initialize(&mut self) {
        for filter_num in 0..self.num_filters {
            self.filters.push(Vec::with_capacity(self.input_channels));
            for channel in 0..self.input_channels {
                let input_size = (self.filter_size*self.filter_size*self.input_channels) as f32;
                let xavier = f32::sqrt(1.0/input_size);
                let mut filter_values:Vec<f32> = Vec::with_capacity(self.filter_size*self.filter_size);
                for filter_index in 0..self.filter_size*self.filter_size {
                    filter_values.push(xavier*(random::<f32>()*2.0 - 1.0));
                }
                self.filters[filter_num].push(Matrix {
                    values: filter_values,
                    rows: self.filter_size,
                    cols: self.filter_size
                });
            }
        }
    }

    fn pass(&self, input: InputTypeEnum<&Vec<Matrix>>) -> OutputTypeEnum<Vec<Matrix>> {
        match input {
            InputTypeEnum::Single(input) => {
                // Flatten filters: from [num_filters x self.input_channels x self.filter_size x self.filter_size]
                let mut flattened_filters = Vec::with_capacity(self.num_filters * self.input_channels * self.filter_size * self.filter_size);
                for f in 0..self.num_filters {
                    for c in 0..self.input_channels {
                        flattened_filters.extend_from_slice(&self.filters[f][c].values);
                    }
                }
                let flattened_filters_matrix = Matrix {
                    values: flattened_filters,
                    rows: self.num_filters,
                    cols: self.input_channels * self.filter_size * self.filter_size,
                };

                // Step 2: Convert input to columns using Im2Col
                let column_matrix = Matrix::im_2_col(input, self.filter_size, self.filter_size, self.stride, self.padding);

                // Step 3: Multiply flattened filters with the column matrix
                let product = flattened_filters_matrix.multiply(&column_matrix);

                // Step 4: Add biases and reshape output
                let mut output = Vec::with_capacity(self.num_filters);
                for filter_idx in 0..self.num_filters {
                    let start_row = filter_idx * self.output_size * self.output_size;
                    let mut sub_matrix = product.sub_matrix(start_row, 0, self.output_size, self.output_size);
                    sub_matrix.add_scalar(self.biases.values[filter_idx]);
                    output.push(sub_matrix);
                }
                OutputTypeEnum::Single(output)
            },
            InputTypeEnum::Batch(inputs) => {
                todo!("Batch pass of convolutional layer.")
            },
        }
    }

    fn backpropagate(&mut self, input: &Vec<Matrix>,
            my_output: &Vec<Matrix>, error: &Vec<Matrix>) -> Vec<Matrix> {

        let mut output_error_values:Vec<Vec<f32>> = vec![];

        for filter_num in 0..self.num_filters {
            let mut bias_gradient = 0.0;
            let mut filter_gradients = zero_matrix(self.filter_size, self.filter_size);

            output_error_values.push(vec![0.0; self.output_size*self.output_size]);

            let current_output = &my_output[filter_num];
            let derivative = activation_derivative(self.activation_function, current_output);
            for x in 0..self.input_size {
                for y in 0..self.input_size {

                    bias_gradient += error[filter_num].get(x, y);

                    for channel in 0..self.input_channels {

                        for fx in 0..self.filter_size {
                            for fy in 0..self.filter_size {
                                let adjusted_x = x + fx - self.padding;
                                let adjusted_y = y + fy - self.padding;
                                let both_positive = x + fx >= self.padding && y + fy >= self.padding;
                                if both_positive && adjusted_x < self.output_size && adjusted_y < self.output_size {
                                    output_error_values[filter_num][x*self.output_size+y] +=
                                            error[filter_num].get(adjusted_x, adjusted_y) *
                                            self.filters[filter_num][channel].get(fx, fy);
                                }

                                if both_positive && adjusted_x < self.input_size && adjusted_y < self.input_size {
                                    let current_input = input[channel].get(adjusted_x, adjusted_y);
                                    let current_derivative = derivative.get(x, y);
                                    filter_gradients.values[fx*self.filter_size+fy] +=
                                            error[filter_num].get(x, y) * current_derivative * current_input;
                                }
                            }
                        }
                    }
                }
            }

            self.biases.values[filter_num] += self.learning_rate * bias_gradient;

            for channel in 0..self.input_channels {
                self.filters[filter_num][channel].add(&filter_gradients);
            }
        }

        let mut output_matrices: Vec<Matrix> = vec![];
        for i in 0..output_error_values.len() {
            output_matrices.push(Matrix {
                values: output_error_values[i].clone(),
                rows: self.input_size,
                cols: self.input_size
            })
        }
        output_matrices
        // let mut dl_dy_values: Vec<f32> = Vec::with_capacity(error.len());
        // for channel in 0..error.len() {
        //     let activation_derivative = activation_derivative(self.activation_function, &my_output[channel]);
        //     let mut dl_dy_channel = error[channel].copy();
        //     dl_dy_channel.element_multiply(&activation_derivative);
        //     dl_dy_values.extend_from_slice(&dl_dy_channel.values[0..dl_dy_channel.values.len()]);
        // }

        // let mut dl_dy = Matrix {
        //     values: dl_dy_values,
        //     rows: self.num_filters,
        //     cols: self.output_size*self.output_size
        // };

        // dl_dy = dl_dy.transpose();

        // let dl_dx = self.filters.transpose().multiply(&dl_dy);

        // let input_to_column = Matrix::im_2_col(input, self.filter_size, self.filter_size, self.stride, self.padding);
        
    }

    fn batch_backpropagate(&mut self, inputs: InputTypeEnum<&Vec<Matrix>>,
            my_outputs: OutputTypeEnum<&Vec<Matrix>>,
            errors: OutputTypeEnum<&Vec<Matrix>>) -> InputTypeEnum<Vec<Matrix>> {
        todo!()
    }

    fn get_input_type_id(&self) -> std::any::TypeId {
        todo!()
    }

    fn get_output_type_id(&self) -> std::any::TypeId {
        todo!()
    }

    fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }
}
/*
match inputs_enum {
            InputTypeEnum::Single(input) => {

            }
            InputTypeEnum::Batch(inputs) => {

            }
        }
*/