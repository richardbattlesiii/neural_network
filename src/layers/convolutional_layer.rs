use crate::helpers::{activation_functions::*, fft};
use crate::helpers::fft::convolve;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use ndarray::{s, Array1, Array2, Array4, ArrayView2, ArrayView4, Axis};

pub const PADDING_VALID:u8 = 0;
pub const PADDING_SAME:u8 = 1;
pub const PADDING_FULL:u8 = 2;

pub struct ConvolutionalLayer {
    pub image_size:usize,
    pub input_channels:usize,

    pub learning_rate:f32,
    pub activation_function:u8,

    filters:Array4<f32>,
    biases:Array1<f32>,
    pub num_filters: usize,
    pub filter_size: usize,
}

impl ConvolutionalLayer {
    pub fn new(
            image_size: usize,
            input_channels: usize,
            num_filters: usize,
            filter_size: usize,
            learning_rate: f32,
            activation_function: u8,
            ) -> ConvolutionalLayer {

        ConvolutionalLayer {
            image_size,
            input_channels,
            learning_rate,
            activation_function,
            filters: Array4::zeros((num_filters, input_channels, filter_size, filter_size)),
            biases: Array1::zeros(num_filters),
            num_filters,
            filter_size,
        }
    }

    pub fn initialize(&mut self) {
        let filter_units = self.filter_size*self.filter_size;
        let input_units = self.input_channels*filter_units;
        let output_units = self.num_filters*filter_units;
        let xavier = (6f32 / (input_units + output_units) as f32).sqrt();
        let filter_distribution = Uniform::new(-xavier, xavier);
        let rng = &mut rand::thread_rng();
        self.filters.map_inplace(|x| *x = filter_distribution.sample(rng));
    }

    ///Note: This is a reference to Longmont Potion Castle, not Rick & Morty.
    pub fn im_ready_to_pass_regular_rick(&self, input: &ArrayView4<f32>) -> Array4<f32> {
        let (batch_size, channels, image_size, _) = input.dim();
        let mut output = Array4::zeros((batch_size, self.num_filters, image_size, image_size));

        for sample in 0..batch_size {
            let mut output_sample = output.index_axis_mut(Axis(0), sample);
            let input_sample = input.index_axis(Axis(0), sample);

            for filter_num in 0..self.num_filters {
                let mut output_filter = output_sample.index_axis_mut(Axis(0), filter_num);
                let filter = self.filters.index_axis(Axis(0), filter_num);

                for channel in 0..channels {
                    let input_channel = input_sample.index_axis(Axis(0), channel);
                    let filter_channel = filter.index_axis(Axis(0), channel);
                    let convolution = convolve_and_slide(&input_channel, &filter_channel, PADDING_SAME);
                    output_filter += &convolution.to_shape((image_size, image_size)).unwrap();
                }
            }
        }

        output
    }

    pub fn backpropagate(&mut self, input: &ArrayView4<f32>,
            my_output: &ArrayView4<f32>, error: &ArrayView4<f32>) -> Array4<f32> {
        let (num_samples, input_channels, image_size, _) = input.dim();
        //dLoss/dFilters
        let mut filter_gradients: Array4<f32> = Array4::zeros((self.num_filters, input_channels, self.filter_size, self.filter_size));
        //dLoss/dBiases
        let mut bias_gradients: Array1<f32> = Array1::zeros(self.num_filters);
        //dLoss/dInputs
        let mut output: Array4<f32> = Array4::zeros((num_samples, input_channels, image_size, image_size));
        //println!("Filters: {:?}", self.filters.shape());

        for sample in 0..num_samples {
            let sample_output = my_output.index_axis(Axis(0), sample);
            let sample_error = error.index_axis(Axis(0), sample);
            let sample_input = input.index_axis(Axis(0), sample);

            for filter_num in 0..self.num_filters {
                //Calculate gradient of loss with respect to the non-activated output
                let filter = self.filters.index_axis(Axis(0), filter_num);
                let current_output = sample_output.index_axis(Axis(0), filter_num);
                let mut derivative = current_output.to_owned();
                activation_derivative_2d(self.activation_function, &mut derivative);

                let current_error = sample_error.index_axis(Axis(0), filter_num);
                let dl_do = &current_error * &derivative;
                bias_gradients[[filter_num]] += dl_do.sum();

                for channel in 0..input_channels {
                    let current_input = sample_input.index_axis(Axis(0), channel);
                    let current_gradients = calculate_filter_gradients(&current_input, &dl_do.view(), self.filter_size);
                    filter_gradients.slice_mut(s![filter_num, channel, .., ..]).assign(&current_gradients);
                    
                    let current_filter = filter.index_axis(Axis(0), channel);
                    let flipped = current_filter.slice(s![..;-1, ..;-1]);
                    let convolution = convolve_and_slide(&current_input, &flipped, PADDING_SAME);
                    let current_output = convolution.to_shape((image_size, image_size)).unwrap();
                    output.slice_mut(s![sample, channel, .., ..]).assign(&current_output);
                }
            }
        }
        
        let coefficient = -self.learning_rate/num_samples as f32;
        self.filters.scaled_add(coefficient, &filter_gradients);
        self.biases.scaled_add(coefficient, &bias_gradients);
        output
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }

    pub fn get_weights_magnitude(&self) -> f32 {
        (&self.filters * &self.filters).sum().sqrt()
    }
}

pub fn convolve_and_slide(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>, padding_mode: u8) -> Array2<f32> {
    let image_size = input.dim().0;
    let kernel_size = kernel.dim().0;
    let filter_min = -(kernel_size as isize) / 2;
    let mut filter_max = (kernel_size / 2) as isize;
    if kernel_size % 2 == 1 {
        filter_max += 1;
    }
    let (output_size, x_y_min, x_y_max) = 
        match padding_mode {
            PADDING_VALID => {(
                input.nrows() - kernel_size + 1,
                -filter_min - 1,
                image_size as isize - filter_max
                )
            }
            PADDING_SAME => {(
                input.nrows(),
                0,
                image_size as isize)
            }
            PADDING_FULL => {(
                input.nrows() + kernel_size - 1,
                filter_min,
                image_size as isize + filter_max)
            }
            _ => {panic!("Invalid padding mode.")}
        };
    let mut output: Array2<f32> = Array2::zeros((output_size, output_size));
    for x in x_y_min..x_y_max {
        for y in x_y_min..x_y_max {
            let mut sum = 0.0;
            for fx in filter_min..filter_max {
                for fy in filter_min..filter_max {
                    if x + fx >= 0 && x + fx < image_size as isize &&
                            y + fy >= 0 && y + fy < image_size as isize {
                        let image_row = (x + fx) as usize;
                        let image_col = (y + fy) as usize;
                        let filter_row = (fx - filter_min) as usize;
                        let filter_col = (fy - filter_min) as usize;
                        sum +=  input[[image_row, image_col]] * 
                                kernel[[filter_row, filter_col]];
                    }
                }
            }
            match padding_mode {
                PADDING_VALID =>{todo!();}
                PADDING_SAME => {output[[x as usize, y as usize]] = sum;}
                PADDING_FULL => {todo!();}
                _ => panic!("Invalid padding mode.")
            }
        }
    }

    output
}

fn calculate_filter_gradients(input: &ArrayView2<f32>, dl_do: &ArrayView2<f32>, filter_size: usize) -> Array2<f32> {
    let mut gradients: Array2<f32> = Array2::zeros((filter_size, filter_size));

    for x in 0..dl_do.nrows() {
        for y in 0..dl_do.ncols() {
            for fx in 0..filter_size {
                for fy in 0..filter_size {
                    let input_x = (x + fx) as isize - 1;
                    let input_y = (y + fy) as isize - 1;
                    if 0 <= input_x && input_x < input.nrows() as isize && 0 <= input_y && input_y < input.ncols() as isize {
                        gradients[[fx , fy]] += dl_do[[x, y]] * input[[input_x as usize, input_y as usize]];
                    }
                }
            }
        }
    }

    gradients
}

///Aaaaaand it's slower than the basic method. Yippee!!!!!!!!
pub fn im_2_col_convolve(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    let debug = false;
    let input_size = input.nrows();
    let kernel_size = kernel.nrows();
    let padded_input = fft::pad(input, input_size + kernel_size - 1);
    if debug {
        println!("Padded input:\n{}", padded_input);
    }
    // Calculate the number of columns (i.e., patches) in the output
    let output_size = input_size * input_size;
    let kernel_flat_size = kernel_size * kernel_size;

    // Initialize the column_vectors matrix
    let mut column_vectors: Array2<f32> = Array2::zeros((kernel_flat_size, output_size));

    // Iterate over all the patches using `.windows()` and flatten them directly into column_vectors
    for (i, window) in padded_input.windows((kernel_size, kernel_size)).into_iter().enumerate() {
        // Flatten the current window (patch) and assign it to the appropriate column
        column_vectors.column_mut(i).assign(&window.iter().cloned().collect::<Array1<f32>>());
        if debug {
            println!("Column vector for patch {}:\n{}", i, column_vectors.column(i));
        }
    }

    // Flatten the kernel into a row vector
    let flattened_kernel: Array2<f32> = Array2::from_shape_vec(
        (1, kernel_flat_size),
        kernel.iter().cloned().collect::<Vec<f32>>(),
    ).unwrap();

    if debug {
        println!("Flattened kernel:\n{}", flattened_kernel);
    }

    if debug {
        println!("Flattened kernel:\n{}", flattened_kernel);
    }
    flattened_kernel.dot(&column_vectors)
}