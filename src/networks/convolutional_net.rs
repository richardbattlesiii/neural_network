// use crate::{flow::flow_ai::PUZZLE_WIDTH, layers::{self, softmax_layer::SoftmaxLayer}};
// use layers::convolutional_layer::ConvolutionalLayer;
// use layers::dense_layer::DenseLayer;
// use layers::layer::Layer;
// use ndarray::{s, Array2, Array4, ArrayView2, ArrayView4};

// #[derive(Default)]
// /*
//     A neural network with a number of convolutional layers, then a flattening layer, then dense layers.
// */
// pub struct ConvolutionalNet {
//     convolutional_layers: Vec<ConvolutionalLayer>,
//     dense_layers: Vec<DenseLayer>,
//     num_convolutional_layers: usize,
//     num_dense_layers: usize,
// }

// impl ConvolutionalNet {
//     ///Make a new ConvolutionalNet. Note that the convolutional layers are assumed to be square for the time being.
//     pub fn new(image_size: usize, input_channels: usize, num_filters: &[usize], filter_sizes: &[usize], dense_layer_sizes: &[usize], activation_functions: &[u8], convolution_method: u8) -> ConvolutionalNet {
//         let mut convolutional_layers = vec![];
//         let num_convolutional_layers = num_filters.len();
//         convolutional_layers.push(ConvolutionalLayer::new(
//             image_size,
//             input_channels,
//             num_filters[0],
//             filter_sizes[0],
//             0.1,
//             activation_functions[0],
//             convolution_method,
//         ));
//         //println!("{} channels", convolutional_layers[0].input_channels);
//         for cl_num in 1..num_convolutional_layers {
//             //println!("{} channels", num_filters[cl_num-1]);
//             convolutional_layers.push(ConvolutionalLayer::new(
//                 image_size,
//                 num_filters[cl_num-1],
//                 num_filters[cl_num],
//                 filter_sizes[cl_num],
//                 0.1,
//                 activation_functions[cl_num],
//                 convolution_method,
//             ));
//         }

//         let mut dense_layers = vec![];
//         let num_dense_layers = dense_layer_sizes.len()-1;
//         for layer in 0..dense_layer_sizes.len()-1 {
//             dense_layers.push(DenseLayer::new(
//                 //Input size
//                 dense_layer_sizes[layer],
//                 //Output size
//                 dense_layer_sizes[layer+1],
//                 //Default learning rate is 0.1 -- must be updated with set_learning_rate or rose_decay.
//                 0.1,
//                 //Lambda for L2 regularization. Default is 0.01.
//                 0.01,
//                 //Activation function
//                 activation_functions[layer + num_convolutional_layers],
//             ));
//         }

//         ConvolutionalNet {
//             convolutional_layers,
//             dense_layers,
//             num_convolutional_layers,
//             num_dense_layers,
//         }
//     }
    
//     pub fn set_learning_rate(&mut self, rate: f32) {
//         for layer in 0..self.num_convolutional_layers {
//             self.convolutional_layers[layer].set_learning_rate(rate);
//         }
//         for layer in 0..self.num_dense_layers {
//             self.dense_layers[layer].set_learning_rate(rate);
//         }
//     }

//     pub fn initialize(&mut self) {
//         for layer in 0..self.num_convolutional_layers {
//             self.convolutional_layers[layer].initialize();
//         }
//         for layer in 0..self.num_dense_layers {
//             self.dense_layers[layer].initialize();
//         }
//     }

//     //Returns the output from each layer.
//     pub fn forward_pass(&self, input: &ArrayView4<f32>) -> (Vec<Array4<f32>>, Vec<Array2<f32>>) {
//         let debug = false;
//         let mut convolutional_outputs: Vec<Array4<f32>> = vec![];
//         convolutional_outputs.push(input.to_owned());
//         let mut current_output = input.to_owned();
//         for layer_num in 0..self.num_convolutional_layers {
//             let output_clone = current_output.clone();
//             let current_input = &output_clone.view();
//             if debug {
//                 println!("Passing conv layer {} with input {:?}.", layer_num, current_input.shape());
//             }
//             current_output = self.convolutional_layers[layer_num].pass(current_input);
//             if debug {
//                 println!("Output was: {:?}", current_output.shape());
//             }
//             convolutional_outputs.push(current_output.clone());
//         }
//         let batch_size = input.dim().0;
//         let mut current_input = current_output.to_shape((batch_size, current_output.len()/batch_size)).unwrap().to_owned();
//         let mut dense_outputs: Vec<Array2<f32>> = vec![];
//         dense_outputs.push(current_input.to_owned());
//         for layer_num in 0..self.num_dense_layers {
//             if debug {
//                 println!("Passing dense layer {} with input {:?}.", layer_num, current_input.shape());
//             }
//             current_input = self.dense_layers[layer_num].pass(&current_input.view());
//             if debug {
//                 println!("Output was {:?}.", current_input.shape());
//             }
//             dense_outputs.push(current_input.clone());
//         }
//         let final_output = &dense_outputs[self.num_dense_layers];
//         dense_outputs.push(SoftmaxLayer::new().pass(&final_output.view()));
//         (convolutional_outputs, dense_outputs)
//     }
    
    
//     // pub fn batch_pass(&self, input: &Tensor) -> Vec<Tensor> {
//     //     let mut all_outputs = vec![];
//     //     all_outputs.push(input.copy());
//     //     for layer in 0..self.num_layers {
//     //         let passed = self.layers[layer].pass(InputTypeEnum::Single(&all_outputs[layer]));
//     //         match passed {
//     //             OutputTypeEnum::Single(output) => {
//     //                 all_outputs.push(output);
//     //             },
//     //             OutputTypeEnum::Batch(_) => panic!("Got a Vec<Tensor> from a DenseLayer. How did that happen????"),
//     //         }
//     //     }
    
//     //     all_outputs
//     // }
    
//     pub fn predict(&self, input: &ArrayView4<f32>) -> Array2<f32> {
//         let (_, dense_outputs) = self.forward_pass(input);
//         dense_outputs[self.num_dense_layers + 1].to_shape((input.dim().0, self.dense_layers[self.num_dense_layers-1].get_parameters().0.ncols())).unwrap().to_owned()
//     }

//     pub fn back_prop(&mut self, inputs: &ArrayView4<f32>, labels: &ArrayView2<f32>) -> f32 {
//         let debug = false;
//         let (convolutional_outputs, dense_outputs) = self.forward_pass(inputs);
        
//         // Calculate initial error with output layer derivative
//         let mut current_error = dense_outputs[self.num_dense_layers+1].clone();
//         //Calculate the loss by stealing the current error before it's done
//         let output = Self::calculate_bce_loss(&current_error.view(), labels);
//         //Derivative of loss function.
//         current_error -= labels;
        
//         if debug {
//             println!("Squared error: {}", (&current_error*&current_error).sum());
//         }
//         for layer in (0..self.num_dense_layers).rev() {
//             if debug {
//                 println!("Backpropagating dense layer {} with error {:?}.", layer, current_error.shape());
//             }
//             let current_layer = &mut self.dense_layers[layer];
//             // Propagate error through activation derivative
//             current_error = current_layer.backpropagate(&dense_outputs[layer].view(), &dense_outputs[layer+1].view(), &current_error.view());
//             // println!("Total error: {}", current_error.sum());
//         }
        
//         let last_conv_layer = &self.convolutional_layers[self.num_convolutional_layers - 1];
//         let shape = (inputs.dim().0, last_conv_layer.num_filters, last_conv_layer.image_size, last_conv_layer.image_size);
//         let mut current_error = current_error.to_shape(shape).unwrap().to_owned();
//         for layer in (0..self.num_convolutional_layers).rev() {
//             if debug {
//                 println!("Backpropagating conv layer {} with input {:?}.", layer, convolutional_outputs[layer].shape());
//             }
//             let current_layer = &mut self.convolutional_layers[layer];
//             current_error = current_layer.backpropagate(&convolutional_outputs[layer].view(), &convolutional_outputs[layer+1].view(), &current_error.view());
//             // println!("Total error: {}", current_error.sum());
//         }

//         output
//     }
    
//     // // pub fn batch_backpropagate(&mut self, inputs: &Tensor, labels: &Tensor) -> f32 {
//     // //     let all_outputs = self.batch_pass(inputs);
    
//     // //     // Calculate initial error with output layer derivative
//     // //     let mut current_errors = all_outputs[all_outputs.len() - 1].copy();
//     // //     let output = DenseNet::calculate_mse_loss(&current_errors, labels);
//     // //     // current_errors = self.custom_loss_derivative(&current_errors, &labels);
//     // //     current_errors.subtract(&labels); //Derivative of loss function.
//     // //     for layer in (0..self.num_layers).rev() {
//     // //         let current_layer = &mut self.layers[layer];
//     // //         // Propagate error through activation derivative
//     // //         let passed_error = current_layer.batch_backpropagate(InputTypeEnum::Single(&all_outputs[layer]),
//     // //                 OutputTypeEnum::Single(&all_outputs[layer + 1]),
//     // //                 OutputTypeEnum::Single(&current_errors));
//     // //         match passed_error {
//     // //             InputTypeEnum::Single(current_error) => current_errors=current_error,
//     // //             InputTypeEnum::Batch(_) => todo!(),
//     // //         }
//     // //     }
//     // //     output
//     // // }

//     // pub fn add_noise(&mut self, range: f32) {
//     //     for layer in 0..self.num_layers {
//     //         self.layers[layer].add_noise(range);
//     //     }
//     // }
    
//     // pub fn calculate_bce_loss(predictions: &Tensor, labels: &Tensor) -> f32 {
//     //     let mut loss = 0.0;
//     //     for i in 0..labels.rows {
//     //         for j in 0..labels.cols {
//     //             let pred = predictions.values[i*labels.cols + j];
//     //             let label = labels.values[i*labels.cols + j];
//     //             loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
//     //         }
//     //     }
//     //     loss / labels.rows as f32
//     // }
    
//     // pub fn calculate_mse_loss(predictions: &Tensor, labels: &Tensor) -> f32 {
//     //     let mut loss = 0.0;
//     //     for i in 0..labels.rows {
//     //         for j in 0..labels.cols {
//     //             let pred = predictions.values[i*labels.cols + j];
//     //             let label = labels.values[i*labels.cols + j];
//     //             let diff = label - pred;
//     //             loss += (diff)*(diff);
//     //         }
//     //     }
//     //     loss / (labels.values.len() as f32)
//     // }
    
//     // pub fn custom_loss_derivative(&self, predictions: &Tensor, labels: &Tensor) -> Tensor {
//     //     let rows = predictions.rows;
//     //     let cols = predictions.cols;
//     //     let mut output_values = Vec::with_capacity(rows*cols);
//     //     for row in 0..rows {
//     //         for col in 0..cols {
//     //             let prediction = predictions.values[row*cols+col];
//     //             let label = labels.values[row*cols+col];
//     //             if (label == 0.0) ^ (f32::round(prediction) != 0.0) {
//     //                 output_values.push(NEGATIVE_ONE_WRONG_PENALTY*(prediction-label));
//     //             }
//     //             else {
//     //                 output_values.push(prediction-label);
//     //             }
//     //         }
//     //     }
    
//     //     Tensor {
//     //         values: output_values,
//     //         rows,
//     //         cols
//     //     }
//     // }
    
//     // pub fn rose_decay_learning_rate(&mut self, epoch: u32, low: f32, high: f32, oscillate_forever: bool,
//     //         oscillation_coefficient: f32, oscillation_parameter: f32, exponential_parameter: f32) -> f32 {
//     //     let rose_decayed_learning_rate;
//     //     if oscillate_forever {
//     //         let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter) + low;
//     //         rose_decayed_learning_rate = exponential_decay * (oscillation_coefficient * f32::sin(oscillation_parameter * epoch as f32) + low)
//     //         + exponential_decay;
//     //     }
//     //     else {
//     //         let exponential_decay = high * f32::exp(epoch as f32 * exponential_parameter);
//     //         rose_decayed_learning_rate = oscillation_coefficient * exponential_decay*f32::sin(oscillation_parameter * epoch as f32)
//     //         + exponential_decay + low;
//     //     }
//     //     self.set_learning_rate(rose_decayed_learning_rate);
//     //     rose_decayed_learning_rate
//     // }

//     // pub fn exponentially_decay_learning_rate(&mut self, epoch: u32, parameter_one: f32, parameter_two: f32,parameter_three: f32) -> f32 {
//     //     let result = parameter_one*f32::exp(epoch as f32 * parameter_two) + parameter_three;
//     //     self.set_learning_rate(result);
//     //     result
//     // }

//     // pub fn oscillate_learning_rate(&mut self, epoch: u32, parameter_one: f32, parameter_two: f32,parameter_three: f32) -> f32 {
//     //     let result = parameter_one*f32::exp(epoch as f32 * parameter_two) + parameter_three;
//     //     self.set_learning_rate(result);
//     //     result
//     // }

//     // pub fn write_net_params_to_string(&self) -> String {
//     //     let mut output = "".to_string();
//     //     for layer in 0..self.num_layers {
//     //         output += &(self.layers[layer].write_params_to_string()+"\n");
//     //     }

//     //     output
//     // }

//     pub fn calculate_bce_loss(predictions: &ArrayView2<f32>, labels: &ArrayView2<f32>) -> f32 {
//         let mut loss = 0.0;
//         for i in 0..labels.nrows() {
//             for j in 0..labels.ncols() {
//                 let epsilon = 1e-7;
//                 let pred = predictions[[i, j]].clamp(epsilon, 1.0-epsilon);
//                 let label = labels[[i, j]].clamp(epsilon, 1.0-epsilon);
//                 loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
//             }
//         }
//         loss /= crate::flow_ai::COLORS as f32;
//         loss / labels.nrows() as f32
//     }
// }