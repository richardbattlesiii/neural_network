use crate::helpers::matrix::*;
use crate::helpers::activation_functions::*;
use rand::Rng;
use std::fmt;

use super::layer::{Layer, InputTypeEnum, OutputTypeEnum};

pub struct ConvolutionalLayer {
    input_size:usize,
    output_size:usize,
    input_channels:usize,

    pub learning_rate:f32,
    pub activation_function:u8,

    filters:Matrix,
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
            filters: Matrix {
                values: vec![0.0; num_filters*input_channels*filter_size*filter_size],
                rows: num_filters,
                cols: input_channels*filter_size*filter_size,
            },
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
        for _ in 0..self.num_filters {
            for _ in 0..self.input_channels {
                todo!()
            }
        }
    }

    fn pass(&self, input: InputTypeEnum<&Vec<Matrix>>) -> OutputTypeEnum<Vec<Matrix>> {
        match input {
            InputTypeEnum::Single(input) => {
                let column_matrix = Matrix::im_2_col(input, self.filter_size, self.filter_size, self.stride, self.padding);
                let product = self.filters.multiply(&column_matrix);
                let mut output = Vec::with_capacity(self.num_filters);
                for row in 0..product.rows {
                    let mut sub_matrix = product.sub_matrix(row*self.output_size, 0, self.output_size, self.output_size);
                    sub_matrix.add_scalar(self.biases.values[row]);
                    output.push(sub_matrix);
                }
                OutputTypeEnum::Single(output)
            },
            InputTypeEnum::Batch(inputs) => {
                let mut output = Vec::with_capacity(inputs.len());
                for sample in 0..inputs.len() {
                    let input = inputs[sample];
                    let column_matrix = Matrix::im_2_col(input, self.filter_size, self.filter_size, self.stride, self.padding);
                    let product = self.filters.multiply(&column_matrix);
                    let mut sample_output = Vec::with_capacity(self.num_filters);
                    for row in 0..product.rows {
                        let mut sub_matrix = product.sub_matrix(row*self.output_size, 0, self.output_size, self.output_size);
                        sub_matrix.add_scalar(self.biases.values[row]);
                        sample_output.push(sub_matrix);
                    }
                    output.push(sample_output);
                }
                OutputTypeEnum::Batch(output)
            },
        }
    }

    fn backpropagate(&mut self, input: &Vec<Matrix>,
            my_output: &Vec<Matrix>, error: &Vec<Matrix>) -> Vec<Matrix> {
                
        todo!()
    }

    fn batch_backpropagate(&mut self, inputs: super::layer::InputTypeEnum<&Vec<Matrix>>,
            my_outputs: super::layer::OutputTypeEnum<&Vec<Matrix>>,
            errors: super::layer::OutputTypeEnum<&Vec<Matrix>>) -> InputTypeEnum<Vec<Matrix>> {
        todo!()
    }

    fn get_input_type_id(&self) -> std::any::TypeId {
        todo!()
    }

    fn get_output_type_id(&self) -> std::any::TypeId {
        todo!()
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