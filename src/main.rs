#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod helpers;
pub mod layers;
pub mod flow;
pub mod networks;
pub mod everhood;
pub mod prelude;

use mnist::Mnist;
use networks::neural_net::{self, calculate_bce_loss, NeuralNet};
use prelude::*;
use std::{f32::consts::TAU, time::Instant};
use flow::flow_ai::{self, COLORS, PUZZLE_WIDTH};
use ndarray::{prelude::*, Slice};

/**
    Used in a custom loss derivative function.
    Because I didn't like that the net was incorrectly getting -1s wrong in the puzzle,
    even though they're literally 1:1 with the input and output. So it just multiplies the
    loss by this value if either the label or the prediction is -1 (not when both are true obviously).
*/
const NEGATIVE_ONE_WRONG_PENALTY: f32 = 2.0;

/**
    For genetic algorithm.
    Corresponds to twice the maximum percentage increase or decrease in each weight and bias.
    I think ideally you would have thread 0 (the one with no mutation) showing up and producing
    a new best about half of the time.
*/
static THREAD_ITERATION_NOISE_RANGE:f32 = 0.4;

///Size of the input and output layers. Since it's one-hot encoded now,
///it's equal to the channels * the size of the puzzle.
const IO_SIZE:usize = COLORS*PUZZLE_WIDTH*PUZZLE_WIDTH;

//Variables used in make_regular_dense_net for comparison:

///Rose Decay. See formula below this variable in main.rs.
const ROSE_DECAY_LEARNING_RATE: u8 = 0;
/*
    Hyperparameters for Rose Decay.

    Formula depends on ROSE_DECAY_OSCILLATE_FOREVER.
    If true:
        exponential_decay = high_value * e^(X*epoch) + low_value
        output = exponential_decay * (o * sin(S*epoch) + low_value)
        - Approaches low*sin(S*epoch) + low*low
    If false:
        exponential_decay = high_value * e^(X*epoch)
        output = o * exponential_decay * sin(S * epoch) + exponential_decay + low_value
        - Approaches low_value, with oscillations decreasing to 0 over time.
*/

///Which version of the formula to use. If true, then it does what it says. It oscillates forever.
const ROSE_DECAY_OSCILLATE_FOREVER: bool = true;

//Corresponds to the X in the formula.
///This should be negative so that the learning rate decays instead of
///increasing exponentially. That would be bad. I tested it to make sure.
const ROSE_DECAY_EXPONENTIAL_PARAMETER: f32 = -1.0/1000.0;

//Corresponds to S in the formula.
///Frequency of oscillations (times tau (which is 2*pi)).
const ROSE_DECAY_OSCILLATION_PARAMETER: f32 = TAU/1000.0;

//Corresponds to o in the formula.
///How big the oscillations are.
const ROSE_DECAY_OSCILLATION_COEFFICIENT: f32 = 0.5;

//Corresponds to high_value in the formula.
///Note that it is NOT the max, it's just relatively high.
const ROSE_DECAY_HIGH_LEARNING_RATE:f32 = 0.1;

//Corresponds to low_value in the formula.
///Not actually the minimum when OSCILLATE_FOREVER is true, just relatively low.
///(It *is* the min when OSCILLATE_FOREVER is false.)
const ROSE_DECAY_LOW_LEARNING_RATE:f32 = 0.05;

///Exponential Decay. learning_rate = p1*e^(epoch*p2)+p3
const EXPONENTIALLY_DECAY_LEARNING_RATE: u8 = 1;
///Exponential Decay. learning_rate = p1*e^(epoch*p2)+p3
const EXPONENTIAL_DECAY_PARAMETER_ONE: f32 = 0.5;
///Exponential Decay. learning_rate = p1*e^(epoch*p2)+p3
const EXPONENTIAL_DECAY_PARAMETER_TWO: f32 = -0.01;
///Exponential Decay. learning_rate = p1*e^(epoch*p2)+p3
const EXPONENTIAL_DECAY_PARAMETER_THREE: f32 = 0.01;

///Oscillate the learning rate. learning_rate = p1 * sin(p2 * epoch) + p3
const OSCILLATE_LEARNING_RATE: u8 = 2;
///Oscillate the learning rate. learning_rate = p1 * sin(p2 * epoch) + p3
const OSCILLATION_PARAMETER_ONE: f32 = -0.01;
///Oscillate the learning rate. learning_rate = p1 * sin(p2 * epoch) + p3
const OSCILLATION_PARAMETER_TWO: f32 = -0.01;
///Oscillate the learning rate. learning_rate = p1 * sin(p2 * epoch) + p3
const OSCILLATION_PARAMETER_THREE: f32 = -0.01;

///Just a static learning rate.
const STATIC_LEARNING_RATE: u8 = 3;
const STATIC_LEARNING_RATE_AMOUNT: f32 = 0.1;

///Used in L2 Regularization.
const LAMBDA: f32 = 0.2;
/**
    Number of threads to use in genetic algorithm. Note that my cpu has 32 threads but that is... atypical.
    So if you run this make sure you set it to something reasonable.
*/
static NUM_THREADS:u8 = 12;

///Number of layers, including input and output layers.
///Number of dense layers will be this minus one.
const NUMBER_OF_LAYERS:usize = 5;

///The size of each layer. Has to be a const for ease of multithreading.
const LAYER_SIZES: [usize; NUMBER_OF_LAYERS] = [IO_SIZE, IO_SIZE/4, IO_SIZE/8, IO_SIZE/4, IO_SIZE];

///The activation function each layer should use. Has to be a const for ease of multithreading.
const ACTIVATION_FUNCTIONS: [u8; NUMBER_OF_LAYERS-1] = [0, 0, 0, 0];

///Number of iterations of the genetic algorithm.
static NUM_TRIES:u32 = 1000000000;

/**
    For genetic algorithm. How much should the number of epochs increase each generation?
    If you're going through iterations quickly, I recommend either 0 or 1. If it's slow, then do whatever you want.
    I'm not your dad.

    Also, this is what makes the genetic algorithm have weird progress percentages. It can't be helped
    (unless I decide to just round the percentages and lie to you in the process).
*/
static EPOCH_INCREASE:u32 = 1;

/**
    Printing interval -- number of epochs between status updates.
    Not used in genetic algorithm.
*/
const PRINTERVAL:u32 = 10;

///Max number of epochs.
///Note that in the genetic algorithm, this is per generation.
static MAX_EPOCHS:u32 = 1000000000;

///How many puzzles to train on.
///Will panic if the total number of puzzles is above the number of lines in the input text file (see flow_ai).
static NUM_TRAINING_PUZZLES:usize = 64;

///How many puzzles to test on.
///Will panic if the total number of puzzles is above the number of lines in the input text file (see flow_ai).
static NUM_TESTING_PUZZLES:usize = 256;

///How often to regenerate the puzzles.
static REGENERATE_PUZZLES_INTERVAL:u32 = 1;

///How many times the genetic algorithm should print a progress update each generation.
const NUM_PRINTS_PER_GENERATION:u32 = 10;

//best testing BCE loss: 1.321
//best confidence in right answer: over 80%
//both with 2000 epochs, 4x4 grid I think

fn main() {
    test_on_mnist();
    // let start = Instant::now();
    // let max = 100000;
    // for i in 0..max {
    //     if i % (max / 100) == 0 {
    //         println!("{}%", i*100/max);
    //     }
    //     flow_ai::generate_solution_with_edging();
    // }
    // println!("Came in {:8.6}s.", start.elapsed().as_secs_f32());

    // let start = Instant::now();
    // flow_ai::generate_puzzles_3d(1000, NUM_THREADS);
    // println!("Finished regular in {:8.6}s.", start.elapsed().as_secs_f32());
    //make_regular_dense_net(ROSE_DECAY_LEARNING_RATE, (0.0, 0.0, 0.0));
    //xor();
    //make_convolutional_net();
    //genetic_algorithm();

    //gamer_net::make_gamer_net();
    // let image = Array2::from_shape_vec((3,3), vec![1.,2.,3.,4.,5.,6.,7.,8.,9.]).unwrap();
    // let kernel = Array2::from_shape_vec((3,3), vec![0.,0.,0.,0.,1.,1.,0.,0.,0.,]).unwrap();
    // println!("Image:\n{}\n\nKernel:\n{}\n", image, kernel,);
    // println!("Result:\n{}", convolutional_layer::im_2_col_convolve(&image.to_owned(), &kernel.to_owned()));
}

const TRAINING_SIZE: usize = 60_000;
const TESTING_SIZE: usize = 10_000;
fn test_on_mnist() {
    let Mnist {
        trn_img, trn_lbl, tst_img, tst_lbl, ..
    } = mnist::MnistBuilder::new()
            .label_format_digit()
            .training_set_length(TRAINING_SIZE as u32)
            .test_set_length(TESTING_SIZE as u32)
            .finalize();

    let train_images = Array4::from_shape_vec(
            (TRAINING_SIZE, 1, 28, 28),
            trn_img.into_iter().map(|x| x as f32 / 255.0).collect(),
    ).unwrap();

    let test_images = Array4::from_shape_vec(
            (TESTING_SIZE, 1, 28, 28),
            tst_img.into_iter().map(|x| x as f32 / 255.0).collect(),
    ).unwrap();

    let train_labels = one_hot_encode(trn_lbl, 10);
    let test_labels = one_hot_encode(tst_lbl, 10);

    let mut net = NeuralNet::new();

    let filters = 16;
    net.add_layer(Box::from(ConvolutionalLayer::new(
        28, //Input size
        1, //Input channels
        filters, //Filters
        5, //Kernel size
        0.4, //Learning rate
        0.01, //Lambda
        RELU, //Activation function
        CONVOLUTION_BASIC //Convolution type
    )));

    net.add_layer(Box::from(ReshapingLayer::new(vec![filters, 28, 28], vec![filters*28*28])));
    net.add_layer(Box::from(DenseLayer::new(
        filters*28*28, //Input size
        512, //Output size
        0.4, //Learning rate
        0.01, //Lambda
        RELU, //Activation Function
    )));

    net.add_layer(Box::from(DenseLayer::new(
        512, //Input size
        10, //Output size
        0.4, //Learning rate
        0.01, //Lambda
        RELU, //Activation Function
    )));

    net.add_layer(Box::from(SoftmaxLayer::new(1, 10)));

    net.initialize();

    let start = Instant::now();
    for epoch in 0..100_000_000 {
        let training_error = net.backpropagate(&train_images.to_owned().into_dyn(), &train_labels.to_owned().into_dyn(), 10);
        let test_predictions = net.predict(&test_images.to_owned().into_dyn());
        let testing_error = neural_net::calculate_bce_loss(&test_predictions, &test_labels.to_owned().into_dyn(), 10);
        println!("Epoch: {epoch},\tTraining: {training_error:8.6}\tTesting: {testing_error:8.6}\tAccuracy: {:5.2}%", accuracy(&test_labels, &test_predictions, 10)*100.);
    }
    let elapsed = start.elapsed().as_millis();
    println!("Finished in {elapsed}ms.");
}

fn accuracy(labels: &Array2<f32>, predictions: &ArrayD<f32>, channels: usize) -> f32 {
    let pred = predictions.to_shape(labels.raw_dim()).unwrap();
    let mut correct = 0;
    let mut total = 0;
    for example in 0..labels.nrows() {
        for i in (0..labels.ncols()).step_by(channels) {
            total += 1;
            let mut max = 0.;
            let mut max_index = -1;
            let mut correct_index = -1;
            for channel in 0..channels {
                if labels[[example, i + channel]] == 1. {
                    if correct_index != -1 {
                        panic!("More than one correct index?");
                    }
                    correct_index = channel as isize;
                }
                if pred[[example, i + channel]] > max {
                    max_index = channel as isize;
                    max = pred[[example, i+channel]];
                }
            }
            if max_index == correct_index {
                correct += 1;
            }
        }
    }

    return correct as f32 / total as f32;
}

fn one_hot_encode(labels: Vec<u8>, num_classes: usize) -> Array2<f32> {
    let num_samples = labels.len();
    let mut one_hot = Array2::<f32>::zeros((num_samples, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[(i, label as usize)] = 1.0;
    }
    one_hot
}

fn make_generic_net() {
    let mut net = NeuralNet::new();

    net.add_layer(Box::from(DropoutLayer::new(
        vec![COLORS, PUZZLE_WIDTH, PUZZLE_WIDTH],
        0.05,
        0.1,
        DROPOUT_MULTIPLY,
    )));

    let channels = &[COLORS, 32, 64];
    let sizes = &[3, 3];
    for i in 0..sizes.len() {
        net.add_layer(Box::from(ConvolutionalLayer::new(
            PUZZLE_WIDTH,
            channels[i],
            channels[i+1],
            sizes[i],
            0.1,
            LAMBDA,
            RELU,
            CONVOLUTION_BASIC
        )))
    }

    let conv_layer_output = vec![channels[channels.len()-1], PUZZLE_WIDTH, PUZZLE_WIDTH];
    let dense_sizes = &[channels[channels.len()-1]*PUZZLE_WIDTH*PUZZLE_WIDTH, IO_SIZE/2, IO_SIZE/2, IO_SIZE/2, IO_SIZE];
    net.add_layer(Box::from(ReshapingLayer::new(
        conv_layer_output,
        vec![dense_sizes[0]]
    )));

    for i in 0..dense_sizes.len()-1 {
        net.add_layer(Box::from(DenseLayer::new(
            dense_sizes[i],
            dense_sizes[i+1],
            0.1,
            LAMBDA,
            RELU
        )));
    }

    net.add_layer(Box::from(SoftmaxLayer::new(PUZZLE_WIDTH*PUZZLE_WIDTH, COLORS)));



    //START TRAINING


    let (mut training_puzzles, mut training_solutions) = flow_ai::generate_puzzles_3d(NUM_TRAINING_PUZZLES, NUM_THREADS, false);
    let (testing_puzzles, testing_solutions) = flow_ai::generate_puzzles_3d(NUM_TESTING_PUZZLES, NUM_THREADS, false);
    println!("Starting training.");
    let start = Instant::now();
    for epoch in 1..MAX_EPOCHS+1 {
        if epoch % REGENERATE_PUZZLES_INTERVAL == 0 {
            (training_puzzles, training_solutions) = flow_ai::generate_puzzles_3d(NUM_TRAINING_PUZZLES, NUM_THREADS, false);
        }

        //println!("training puzzles are {:?}", training_puzzles.shape());
        let training_loss = net.backpropagate(&training_puzzles.to_owned().into_dyn(), &training_solutions.to_owned().into_dyn(), COLORS);

        if epoch % PRINTERVAL == 0 {
            let testing_loss = calculate_bce_loss(&net.predict(&testing_puzzles.to_owned().into_dyn()).to_owned(), &testing_solutions.to_owned().into_dyn(), COLORS);
            println!("Epoch {} -- training loss: {:8.6}, testing loss: {:8.6}", epoch, training_loss, testing_loss);
            
            if epoch % (PRINTERVAL * 10) == 0 {
                test_net_specific(&net, &testing_puzzles.to_owned().into_dyn(), &testing_solutions.to_owned());
            }
        }
    }

    println!("Finished in {}ms.", start.elapsed().as_millis());
}



///Make a DenseNet with the specified learning rate change method -- 
///Rose Decay, exponential decay, oscillation, static. Panics if invalid.
// fn make_regular_dense_net(learning_rate_change_method: u8, parameters: (f32, f32, f32)) -> f32 {
//     //Get a Matrix for the puzzles and their solutions.
//     //Shape is [number of puzzles X IO_SIZE]
//     let ( mut training_puzzles, mut training_solutions) = flow_ai::generate_puzzles_1d(NUM_TRAINING_PUZZLES);
//     let (testing_puzzles, testing_solutions) = flow_ai::generate_puzzles_1d(NUM_TESTING_PUZZLES);
    
//     //Make a DenseNet with the constant layer sizes & activation functions.
//     let mut dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
//     //Randomize the weights using xavier initialization.
//     dn.initialize();
//     dn.set_lambda(LAMBDA);
//     //test_net_specific(&dn, &puzzles, &solutions);
    
//     //Keep track of how well the network is doing
//     let mut best_loss = f32::MAX;
//     let mut best_epoch = 0;
//     for epoch in 0..MAX_EPOCHS+1 {
//         //Regenerate the puzzles every once in a while because it doesn't take that long.
//         if epoch % REGENERATE_PUZZLES_INTERVAL == 0 && epoch > 0 {
//             (training_puzzles, training_solutions) = flow_ai::generate_puzzles_1d(NUM_TRAINING_PUZZLES);
//             // (testing_puzzles, testing_solutions) = (training_puzzles.clone(), training_solutions.clone());
//         }
        
//         //Use the specified learning rate change method to update the learning rate, and print out what it is.
//         // let learning_rate;
//         match learning_rate_change_method {
//             ROSE_DECAY_LEARNING_RATE => {
//                 dn.rose_decay_learning_rate(epoch, ROSE_DECAY_LOW_LEARNING_RATE,
//                         ROSE_DECAY_HIGH_LEARNING_RATE, ROSE_DECAY_OSCILLATE_FOREVER,
//                         ROSE_DECAY_OSCILLATION_COEFFICIENT, ROSE_DECAY_OSCILLATION_PARAMETER, ROSE_DECAY_EXPONENTIAL_PARAMETER);
                
//                 // if epoch % PRINTERVAL == 0 {
//                 //     println!("\tLearning rate for Rose Decay is {}.", learning_rate);
//                 // }
//             },
//             EXPONENTIALLY_DECAY_LEARNING_RATE => {
//                 dn.exponentially_decay_learning_rate(epoch,
//                         parameters.0, parameters.1, parameters.2);
//                 // if epoch % PRINTERVAL == 0 {   
//                 //     println!("\tLearning rate for exponential decay is {}.", learning_rate);
//                 // }
//             },
//             OSCILLATE_LEARNING_RATE => {
//                 dn.oscillate_learning_rate(epoch,
//                     parameters.0, parameters.1, parameters.2);
//                 // if epoch % PRINTERVAL == 0 {
//                 //     println!("\tLearning rate for oscillation is {}.", learning_rate);
//                 // }
//             },
//             STATIC_LEARNING_RATE => {
//                 dn.set_learning_rate(parameters.0);
//             },
//             default => panic!("Invalid learning rate change method."),
//         }

//         //Train the net.
//         let training_loss = dn.backpropagate(&training_puzzles.to_owned(), &training_solutions.to_owned());

//         //Print the progress, finding the MSE of predictions on the test puzzles
//         if epoch % PRINTERVAL == 0 {
//             let average_loss = test_dense_net(&dn, &testing_puzzles, &testing_solutions);
//             println!("Change method: {}, Epoch: {},\tTesting Loss: {:8.4},  \tTraining Loss: {:8.4}", learning_rate_change_method, epoch, average_loss, training_loss);
//             if epoch % (PRINTERVAL*10) == 0 {
//                 test_net_specific(&dn, &testing_puzzles, &testing_solutions);
//             }
//             if average_loss < best_loss {
//                 best_loss = average_loss;
//                 best_epoch = epoch;
//             }
//             else if average_loss > best_loss*1.05 &&
//                     epoch as f32 > best_epoch as f32+REGENERATE_PUZZLES_INTERVAL as f32 * 1.1 {
//                 // println!("Best was epoch {} with {} loss.", best_epoch, best_loss);
//                 // println!("Overfitting, damn it...");
//                 // break;
//             }
//         }

        
//     }
//     test_net_specific(&dn, &testing_puzzles, &testing_solutions);
//     test_dense_net(&dn, &testing_puzzles, &testing_solutions)

//     // //Print how long it took.
//     // let duration = start.elapsed().as_millis();
//     // println!("Finished in {}ms.", duration);
// }

// //Uses multiple threads, each training the same neural net but with noise added
// //NOTE: technically this involves "training" on the test set because of the way it selects the best network,
// //but it's so indirect that I simply do not care for the time being.
// fn genetic_algorithm() {
//     //Get a matrix from flow_ai (using a pre-made text file from Java)
//     //TODO: rewrite that functionality in Rust
//     let (puzzles, solutions) = convert().unwrap();
    
//     let puzzle_tuple_original = Arc::new(Mutex::new((puzzles, solutions)));

//     //Make a neural net with default values, and say its loss was f32:MAX
//     let worst = DenseNet::default();
//     let best = Arc::new(Mutex::new((worst, f32::MAX)));
//     //Keep track of when the best actually changes between iterations
//     let mut current_best = f32::MAX;
//     for i in 0..NUM_TRIES {
//         //Create a vector to hold the handles of the spawned threads
//         let mut handles = vec![];

//         for thread_num in 0..NUM_THREADS {
//             let best = Arc::clone(&best);
//             let puzzle_tuple = Arc::clone(&puzzle_tuple_original);
            
//             //Spawn a new thread and train a neural net
//             let handle = thread::spawn(move || {
//                 //I like having some idea of the progress, this was the easiest way to do that.
//                 if thread_num == 0 {
//                     println!("Starting generation {}...", i+1);
//                 }
//                 let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();

//                 //Get the training puzzles & their solutions from the master tuple
//                 let puzzles = puzzle_tuple_thread.0.slice(s![0..NUM_TRAINING_PUZZLES, 0..IO_SIZE]).to_owned();
//                 let solutions = puzzle_tuple_thread.1.slice(s![0..NUM_TRAINING_PUZZLES, 0..IO_SIZE]).to_owned();
//                 //Don't need that any more
//                 drop(puzzle_tuple_thread);

//                 let mut dn;
//                 //On the first generation, don't even bother using 'worst', just make your own.
//                 if i == 0 {
//                     dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
//                     dn.initialize();
//                 }
//                 else {
//                     //On subsequent iterations, take the best one
//                     let best_thread = best.lock().unwrap();
//                     dn = best_thread.0.clone();
//                     //And add some noise, analagous to a mutation in a genetic algorithm
//                     if thread_num != 0 {
//                         dn.add_noise(THREAD_ITERATION_NOISE_RANGE);
//                     }
//                 }

//                 let mut result = -1.0;
//                 //Now actually train it, with the number of iterations
//                 //increasing the number of epochs by EPOCH_INCREASE
//                 for epoch in 0..MAX_EPOCHS+i*EPOCH_INCREASE {
//                     //For thread 0, print progress updates an amount of times equal to NUM_PRINTS_PER_GENERATION
//                     if thread_num == 0 && epoch % 
//                             ((MAX_EPOCHS+EPOCH_INCREASE*i) / NUM_PRINTS_PER_GENERATION)
//                             == 0 {
//                         println!("\t{}%...", epoch*100/(MAX_EPOCHS+EPOCH_INCREASE*i));
//                     }
//                     result = dn.backpropagate(&puzzles.to_owned(), &solutions.to_owned());
//                 }

//                 //Get the test puzzles, same as the training puzzles but starting after the index they ended
//                 let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();
//                 let test_puzzles = puzzle_tuple_thread.0.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
//                 let test_solutions = puzzle_tuple_thread.1.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
//                 drop(puzzle_tuple_thread);
                
//                 //Test the net and find the average loss over the testing puzzles
//                 let final_loss = test_dense_net(&dn, &test_puzzles, &test_solutions);
//                 //Means "variable named 'best' but for this thread" not "the thread that is the best"
//                 let mut best_thread = best.lock().unwrap();
//                 //Check if this thread did a better job, if so, update best and print the results.
//                 if best_thread.1 > final_loss {
//                     best_thread.0 = dn;
//                     best_thread.1 = final_loss;
//                     println!("New best from thread {} -- {}, training loss: {}", thread_num, final_loss, result);
//                 }
//                 //See ya later stinky
//                 drop(best_thread);
                
//             });

//             //Store the handle so we can know when it's done
//             handles.push(handle);
//         }

//         //Wait for all threads to finish
//         for handle in handles {
//             handle.join().unwrap();
//         }

//         //Back in main thread, check if we got a new best.
//         let best_main = best.lock().unwrap();
//         if best_main.1 < current_best {
//             current_best = best_main.1;
//             let puzzle_tuple = puzzle_tuple_original.lock().unwrap();

//             //test_net_specific(&best_main.0, &puzzle_tuple.0, &puzzle_tuple.1);

//             //Keep track of the parameters in a file for later, in case something goes wrong during training.
//             //(or more likely I get impatient or want to use my whole CPU again so I press Ctrl-C)
//             let mut output_file = File::create("net.txt").unwrap();
//             output_file.write_all(best_main.0.write_net_params_to_string().as_bytes()).unwrap();
//         }

//         drop(best_main);
//     }

//     //Now we're done iterating (usually I don't let the program get this far by setting NUM_TRIES really high)
//     let best = &best.lock().unwrap().0;
//     let puzzle_tuple_thread = puzzle_tuple_original.lock().unwrap();
//     let test_puzzles = puzzle_tuple_thread.0.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
//     let test_solutions = puzzle_tuple_thread.1.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
//     //Not needed but doesn't hurt
//     drop(puzzle_tuple_thread);
    
//     let mut final_loss = 0.0;
//     for test_puzzle_num in 0..NUM_TESTING_PUZZLES {
//         let test_puzzle = test_puzzles.slice(s![test_puzzle_num..test_puzzle_num+1, 0..IO_SIZE]).to_owned();
//         let test_prediction = best.predict(&test_puzzle.to_owned());
//         let test_solution = test_solutions.slice(s![test_puzzle_num..test_puzzle_num+1, 0..IO_SIZE]).to_owned();
//         //TODO: change to BCE after switching to one-hot
//         let loss = best.calculate_bce_loss(&test_prediction.to_owned(), &test_solution.to_owned());
//         final_loss += loss;
//     }
//     final_loss /= NUM_TESTING_PUZZLES as f32;
//     println!("Final results: {}", final_loss);

//     // // let a = matrix::rand(4096, 4096, 1.0);
//     // // let b = matrix::rand(4096, 4096, 1.0);
//     // println!("Starting...");
//     // // let start = Instant::now();
//     // // Matrix::merge(&a, &a, &a, &a);

//     // // let duration = start.elapsed();
//     // // println!("Time elapsed: {} ms", duration.as_millis());
// }

// ///Get the average BCE across all the testing puzzles.
// fn test_dense_net(dn: &DenseNet, testing_puzzles: &Array2<f32>, testing_solutions: &Array2<f32>) -> f32 {
//     dn.calculate_bce_loss(&dn.predict(&testing_puzzles.to_owned()).to_owned(), &testing_solutions.to_owned())
// }

///Get a random test puzzle and make a prediction on it.
fn test_net_specific(net: &NeuralNet, puzzles: &ArrayD<f32>, solutions: &Array2<f32>) {
    let randy:f32 = rand::random();
    let puzzle_num = (randy*(puzzles.dim()[0] as f32)) as usize;
    let puzzle = puzzles.slice_axis(Axis(0), Slice::new(puzzle_num as isize, Some(puzzle_num as isize + 1), 1)).to_owned();
    let solution = solutions.slice_axis(Axis(0), Slice::new(puzzle_num as isize, Some(puzzle_num as isize + 1), 1)).to_owned();
    let prediction = net.predict(&puzzle.to_owned().into_dyn());
    // println!("{}", solution);
    // println!("{}", prediction);
    let converted_solution = predict_from_one_hot(&solution.slice(s![0, ..]));
    let converted_prediction = predict_from_one_hot(&prediction.slice(s![0, ..]));
    //Print out the solution vs the prediction of the puzzle,
    //to see how good (or more likely bad) the net really is at Flow Free.
    println!("{}", converted_solution);
    println!("{}", converted_prediction);
    print_confidence_in_right_answer(&prediction.to_shape(solution.raw_dim()).unwrap().to_owned(), &solution.to_owned());
}

///Converts one-hot encoding into a grid of predictions
fn predict_from_one_hot(prediction: &ArrayView1<f32>) -> Array2<f32> {
    let mut output = Array2::zeros((PUZZLE_WIDTH, PUZZLE_WIDTH));
    for row in 0..output.nrows() {
        for col in 0..output.ncols() {
            let mut highest_value = -1.0;
            let mut best_color = -1.0;
            for color in 0..COLORS {
                let pred = prediction[[row*PUZZLE_WIDTH*COLORS + col*COLORS + color]];
                if pred > highest_value {
                    best_color = color as f32;
                    highest_value = pred;
                }
            }
            output[[row, col]] = best_color;
        }
    }

    output
}

/**
    I don't really know how to describe what this is, but I wanted a sequence that
    successively "scans" between 0 and 1 with increasing resolution, meaning
    it starts at 1/2, then 1/4, then 3/4, then 1/8, 3/8, 5/8, 7/8, 1/16, etc.

    Used for automatic hyperparameter optimization.
*/
fn interpolate_by_halves(iterations: u32) -> f32 {
    let mut count = 0;
    let mut n = 1.0;

    //Loop until we reach the iterations-th element in the sequence
    while count < iterations {
        //Iterate over all odd numerators for the current denominator
        for numerator in (1..(2 * n as usize)).step_by(2) {
            count += 1;
            if count == iterations {
                return numerator as f32 / 2.0 / n; //Return the fraction
            }
        }
        n *= 2.0; //Move to the next power of 2
    }

    //In case input is invalid (shouldn't happen but whatever)
    0.0
}

///Prints the average confidence across all correct colors.
fn print_confidence_in_right_answer(prediction: &Array2<f32>, solution: &Array2<f32>) {
    let debug = false;
    //Stores the confidence in the right answers
    let mut confidence = 0.;
    //Loop over the puzzles
    for row in 0..prediction.dim().0 {
        //Loop over the tiles in the puzzle, checking if each one is correct
        for col in 0..prediction.dim().1 {
            if solution[[row, col]] == 1f32 {
                confidence += prediction[[row, col]];
                if debug {
                    println!("Added ({}, {}) which should be tile {} color {}.", row, col, col/COLORS, col%COLORS);
                }
            }
        }
        if debug {
            println!("Total confidence was: {}", confidence);
        }
    }
    //Take the average
    confidence /= (prediction.nrows()*PUZZLE_WIDTH*PUZZLE_WIDTH) as f32;
    println!("Average correct confidence: {:6.5}, That's {:4.2} times better than average!", confidence, confidence*COLORS as f32);
}