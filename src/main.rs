#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

pub mod helpers;
pub mod layers;
pub mod flow;
pub mod networks;

use core::f32;
use std::fs::File;
use std::io::{IoSlice, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use helpers::activation_functions;
use rand::{random, Rng};
use std::f32::consts::TAU;
use std::time::{Instant, Duration};
use flow::flow_ai::{self, convert, COLORS, PUZZLE_WIDTH};
use ndarray::{Array2, s};
use networks::dense_net::DenseNet;

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
/**
    This should be negative so that the learning rate decays instead of
    increasing exponentially. That would be bad. I tested it to make sure.

    Corresponds to the X in the formula.
*/
const ROSE_DECAY_EXPONENTIAL_PARAMETER: f32 = -1.0/10000.0;
//Corresponds to S in the formula.
///Frequency of oscillations (times tau (which is 2*pi)).
const ROSE_DECAY_OSCILLATION_PARAMETER: f32 = TAU/1000.0;
//Corresponds to o in the formula.
///How big the oscillations are.
const ROSE_DECAY_OSCILLATION_COEFFICIENT: f32 = 0.5;
//Corresponds to high_value in the formula.
///Note that it is NOT the max, it's just relatively high.
const ROSE_DECAY_HIGH_LEARNING_RATE:f32 = 0.5;
//Corresponds to low_value in the formula.
///Not actual the minimum when OSCILLATE_FOREVER is true, just relatively low.
///(It *is* the min when OSCILLATE_FOREVER is false.)
const ROSE_DECAY_LOW_LEARNING_RATE:f32 = 0.1;

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
const LAMBDA: f32 = 0.005;
/**
    Number of threads to use in genetic algorithm. Note that my cpu has 32 threads but that is... atypical.
    So if you run this make sure you change this unless you also have a lot of cores.
*/
static NUM_THREADS:u32 = 12;

///Number of layers, including input and output layers.
///Number of dense layers will be this minus one.
const NUMBER_OF_LAYERS:usize = 3;

///The size of each layer. Has to be a const for ease of multithreading.
const LAYER_SIZES: [usize; NUMBER_OF_LAYERS] = [IO_SIZE, 4, IO_SIZE];

///The activation function each layer should use. Has to be a const for ease of multithreading.
const ACTIVATION_FUNCTIONS: [u8; NUMBER_OF_LAYERS-1] = [0, 1];

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
const PRINTERVAL:u32 = 100;

///Max number of epochs.
///Note that in the genetic algorithm, this is per generation.
static MAX_EPOCHS:u32 = 10000000;

///How many puzzles to train on.
///Will panic if the total number of puzzles is above the number of lines in the input text file (see flow_ai).
static NUM_TRAINING_PUZZLES:usize = 16384;

///How many puzzles to test on.
///Will panic if the total number of puzzles is above the number of lines in the input text file (see flow_ai).
static NUM_TESTING_PUZZLES:usize = 1024;

///How often to regenerate the puzzles.
static REGENERATE_PUZZLES_INTERVAL:u32 = 5000;

///How many times the genetic algorithm should print a progress update each generation.
const NUM_PRINTS_PER_GENERATION:u32 = 10;

fn main() {
    //xor();
    make_regular_dense_net(ROSE_DECAY_LEARNING_RATE, (0.0,0.0,0.0));
    //genetic_algorithm();

    // let best_ever_loss = f32::MAX;
    // let best_learning_rate = -1.0;
    // // let mut handles = vec![];
    // for i in 1..NUM_TRIES {
    //     // handles.push(thread::spawn(move || {
    //         let current_learning_rate = interpolate_by_halves(i);
    //         println!("Trying learning rate of {}", current_learning_rate);
    //         let mut best_current_loss = f32::MAX;
    //         for j in 0..10 {
    //             let current_loss = make_regular_dense_net(STATIC_LEARNING_RATE, (current_learning_rate, 0.0, 0.0));
    //             if current_loss < best_current_loss {
    //                 best_current_loss = current_loss;
    //             }
    //         }
    //         println!("Loss was {}", best_current_loss);
    //         if best_current_loss < best_ever_loss {
    //             println!("New best!");

    //         }
    //     //     } ));
        
    //     // if i >= NUM_THREADS {
    //     //     for handle in handles.drain(..NUM_THREADS as usize) {
    //     //         handle.join().unwrap();
    //     //     }
    //     // }
    // }
}

///Make a DenseNet and have it solve XOR, as a minimum working example to compare with working
///code.
fn xor() {
    let inputs:Array2<f32> = Array2::from_shape_vec((4,2), vec![0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0]).unwrap();
    let labels:Array2<f32> = Array2::from_shape_vec((4,1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    let layer_sizes = vec![2, 18, 1];
    let activation_functions = vec![0, 1];
    let mut dn = DenseNet::new_with_vectors(&layer_sizes, &activation_functions);
    dn.initialize();
    dn.set_learning_rate(0.3);
    let mut training_error = -1.0;
    for i in 0..MAX_EPOCHS {
        training_error = dn.backpropagate(&inputs, &labels);
        // for j in 0..4 {
        //     training_error = dn.backpropagate(&inputs.slice(s![j..j+1, 0..2]).to_owned(), &labels.slice(s![j..j+1, 0..2]).to_owned());
        // }
        //println!("Training error: {}", training_error);
    }
    println!("{}", training_error);
    println!("{}", dn.predict(&inputs));
}
///Make a DenseNet with the specified learning rate change method -- 
///Rose Decay, exponential decay, oscillation, static. Panics if invalid.
fn make_regular_dense_net(learning_rate_change_method: u8, parameters: (f32, f32, f32)) -> f32 {
    //Get a Matrix for the puzzles and their solutions.
    //Shape is [number of puzzles X IO_SIZE]
    let (training_puzzles, training_solutions) = flow_ai::generate_puzzles(NUM_TRAINING_PUZZLES);
    let (testing_puzzles, testing_solutions) = flow_ai::generate_puzzles(NUM_TESTING_PUZZLES);
    
    //Make a DenseNet with the constant layer sizes & activation functions.
    let mut dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
    //Randomize the weights using xavier initialization.
    dn.initialize();
    dn.set_lambda(LAMBDA);
    //test_net_specific(&dn, &puzzles, &solutions);
    
    //Keep track of how well the network is doing
    let mut best_loss = f32::MAX;
    let mut best_epoch = 0;
    for epoch in 0..MAX_EPOCHS+1 {
        //Regenerate the puzzles every once in a while because it doesn't take that long.
        if epoch % REGENERATE_PUZZLES_INTERVAL == 0 && epoch > 0 {
            let (training_puzzles, training_solutions) = flow_ai::generate_puzzles(NUM_TRAINING_PUZZLES);
            let (testing_puzzles, testing_solutions) = flow_ai::generate_puzzles(NUM_TESTING_PUZZLES);
        }
        
        //Use the specified learning rate change method to update the learning rate, and print out what it is.
        let learning_rate;
        match learning_rate_change_method {
            ROSE_DECAY_LEARNING_RATE => {
                learning_rate = dn.rose_decay_learning_rate(epoch, ROSE_DECAY_LOW_LEARNING_RATE,
                        ROSE_DECAY_HIGH_LEARNING_RATE, ROSE_DECAY_OSCILLATE_FOREVER,
                        ROSE_DECAY_OSCILLATION_COEFFICIENT, ROSE_DECAY_OSCILLATION_PARAMETER, ROSE_DECAY_EXPONENTIAL_PARAMETER);
                
                if epoch % PRINTERVAL == 0 {
                    println!("\tLearning rate for Rose Decay is {}.", learning_rate);
                }
            },
            EXPONENTIALLY_DECAY_LEARNING_RATE => {
                learning_rate = dn.exponentially_decay_learning_rate(epoch,
                        parameters.0, parameters.1, parameters.2);
                if epoch % PRINTERVAL == 0 {   
                    println!("\tLearning rate for exponential decay is {}.", learning_rate);
                }
            },
            OSCILLATE_LEARNING_RATE => {
                learning_rate = dn.oscillate_learning_rate(epoch,
                    parameters.0, parameters.1, parameters.2);
                if epoch % PRINTERVAL == 0 {
                    println!("\tLearning rate for oscillation is {}.", learning_rate);
                }
            },
            STATIC_LEARNING_RATE => {
                dn.set_learning_rate(parameters.0);
            },
            default => panic!("Invalid learning rate change method."),
        }

        //Train the net.
        let training_loss = dn.backpropagate(&training_puzzles, &training_solutions);
        
        //Sleep so that the threads always print in the same order.
        //std::thread::sleep(Duration::from_secs_f32(learning_rate_change_method as f32));

        //Print the progress, finding the MSE of predictions on the test puzzles
        if epoch % PRINTERVAL == 0 {
            let average_loss = test_net(&dn, &testing_puzzles, &testing_solutions);
            println!("Change method: {}, Epoch: {},\tTesting Loss: {:8.4},  \tTraining Loss: {:8.4}", learning_rate_change_method, epoch, average_loss, training_loss);
            if epoch % (PRINTERVAL*10) == 0 {
                test_net_specific(&dn, &testing_puzzles, &testing_solutions);
            }
            if average_loss < best_loss {
                best_loss = average_loss;
                best_epoch = epoch;
            }
            else if average_loss > best_loss*1.05 &&
                    epoch as f32 > best_epoch as f32+REGENERATE_PUZZLES_INTERVAL as f32 * 1.1 {
                println!("Best was epoch {} with {} loss.", best_epoch, best_loss);
                println!("Overfitting, damn it...");
                break;
            }
        }

        
    }
    test_net_specific(&dn, &testing_puzzles, &testing_solutions);
    test_net(&dn, &testing_puzzles, &testing_solutions)

    // //Print how long it took.
    // let duration = start.elapsed().as_millis();
    // println!("Finished in {}ms.", duration);
}

//Uses multiple threads, each training the same neural net but with noise added
//NOTE: technically this involves "training" on the test set because of the way it selects the best network,
//but it's so indirect that I simply do not care for the time being.
fn genetic_algorithm() {
    //Get a matrix from flow_ai (using a pre-made text file from Java)
    //TODO: rewrite that functionality in Rust
    let (puzzles, solutions) = convert().unwrap();
    
    let puzzle_tuple_original = Arc::new(Mutex::new((puzzles, solutions)));

    //Make a neural net with default values, and say its loss was f32:MAX
    let worst = DenseNet::default();
    let best = Arc::new(Mutex::new((worst, f32::MAX)));
    //Keep track of when the best actually changes between iterations
    let mut current_best = f32::MAX;
    for i in 0..NUM_TRIES {
        //Create a vector to hold the handles of the spawned threads
        let mut handles = vec![];

        for thread_num in 0..NUM_THREADS {
            let best = Arc::clone(&best);
            let puzzle_tuple = Arc::clone(&puzzle_tuple_original);
            
            //Spawn a new thread and train a neural net
            let handle = thread::spawn(move || {
                //I like having some idea of the progress, this was the easiest way to do that.
                if thread_num == 0 {
                    println!("Starting generation {}...", i+1);
                }
                let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();

                //Get the training puzzles & their solutions from the master tuple
                let puzzles = puzzle_tuple_thread.0.slice(s![0..NUM_TRAINING_PUZZLES, 0..IO_SIZE]).to_owned();
                let solutions = puzzle_tuple_thread.1.slice(s![0..NUM_TRAINING_PUZZLES, 0..IO_SIZE]).to_owned();
                //Don't need that any more
                drop(puzzle_tuple_thread);

                let mut dn;
                //On the first generation, don't even bother using 'worst', just make your own.
                if i == 0 {
                    dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
                    dn.initialize();
                }
                else {
                    //On subsequent iterations, take the best one
                    let best_thread = best.lock().unwrap();
                    dn = best_thread.0.clone();
                    //And add some noise, analagous to a mutation in a genetic algorithm
                    if thread_num != 0 {
                        dn.add_noise(THREAD_ITERATION_NOISE_RANGE);
                    }
                }

                let mut result = -1.0;
                //Now actually train it, with the number of iterations
                //increasing the number of epochs by EPOCH_INCREASE
                for epoch in 0..MAX_EPOCHS+i*EPOCH_INCREASE {
                    //For thread 0, print progress updates an amount of times equal to NUM_PRINTS_PER_GENERATION
                    if thread_num == 0 && epoch % 
                            ((MAX_EPOCHS+EPOCH_INCREASE*i) / NUM_PRINTS_PER_GENERATION)
                            == 0 {
                        println!("\t{}%...", epoch*100/(MAX_EPOCHS+EPOCH_INCREASE*i));
                    }
                    result = dn.backpropagate(&puzzles, &solutions);
                }

                //Get the test puzzles, same as the training puzzles but starting after the index they ended
                let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();
                let test_puzzles = puzzle_tuple_thread.0.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
                let test_solutions = puzzle_tuple_thread.1.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
                drop(puzzle_tuple_thread);
                
                //Test the net and find the average loss over the testing puzzles
                let final_loss = test_net(&dn, &test_puzzles, &test_solutions);
                //Means "variable named 'best' but for this thread" not "the thread that is the best"
                let mut best_thread = best.lock().unwrap();
                //Check if this thread did a better job, if so, update best and print the results.
                if best_thread.1 > final_loss {
                    best_thread.0 = dn;
                    best_thread.1 = final_loss;
                    println!("New best from thread {} -- {}, training loss: {}", thread_num, final_loss, result);
                }
                //See ya later stinky
                drop(best_thread);
                
            });

            //Store the handle so we can know when it's done
            handles.push(handle);
        }

        //Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }

        //Back in main thread, check if we got a new best.
        let best_main = best.lock().unwrap();
        if best_main.1 < current_best {
            current_best = best_main.1;
            let puzzle_tuple = puzzle_tuple_original.lock().unwrap();

            //test_net_specific(&best_main.0, &puzzle_tuple.0, &puzzle_tuple.1);

            //Keep track of the parameters in a file for later, in case something goes wrong during training.
            //(or more likely I get impatient or want to use my whole CPU again so I press Ctrl-C)
            let mut output_file = File::create("net.txt").unwrap();
            output_file.write_all(best_main.0.write_net_params_to_string().as_bytes()).unwrap();
        }

        drop(best_main);
    }

    //Now we're done iterating (usually I don't let the program get this far by setting NUM_TRIES really high)
    let best = &best.lock().unwrap().0;
    let puzzle_tuple_thread = puzzle_tuple_original.lock().unwrap();
    let test_puzzles = puzzle_tuple_thread.0.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
    let test_solutions = puzzle_tuple_thread.1.slice(s![NUM_TRAINING_PUZZLES..NUM_TRAINING_PUZZLES+NUM_TESTING_PUZZLES, 0..IO_SIZE]).to_owned();
    //Not needed but doesn't hurt
    drop(puzzle_tuple_thread);
    
    let mut final_loss = 0.0;
    for test_puzzle_num in 0..NUM_TESTING_PUZZLES {
        let test_puzzle = test_puzzles.slice(s![test_puzzle_num..test_puzzle_num+1, 0..IO_SIZE]).to_owned();
        let test_prediction = best.predict(&test_puzzle);
        let test_solution = test_solutions.slice(s![test_puzzle_num..test_puzzle_num+1, 0..IO_SIZE]).to_owned();
        //TODO: change to BCE after switching to one-hot
        let loss = best.calculate_bce_loss(&test_prediction, &test_solution);
        final_loss += loss;
    }
    final_loss /= NUM_TESTING_PUZZLES as f32;
    println!("Final results: {}", final_loss);

    // // let a = matrix::rand(4096, 4096, 1.0);
    // // let b = matrix::rand(4096, 4096, 1.0);
    // println!("Starting...");
    // // let start = Instant::now();
    // // Matrix::merge(&a, &a, &a, &a);

    // // let duration = start.elapsed();
    // // println!("Time elapsed: {} ms", duration.as_millis());
}

fn test_net(dn: &DenseNet, testing_puzzles: &Array2<f32>, testing_solutions: &Array2<f32>) -> f32 {
    dn.calculate_bce_loss(&dn.predict(testing_puzzles), testing_solutions)
}

fn test_net_specific(dn: &DenseNet, puzzles: &Array2<f32>, solutions: &Array2<f32>) {
    //Get a random test puzzle and make a prediction on it.
    let randy:f32 = random();
    let puzzle_num = (randy*(puzzles.nrows() as f32)) as usize;
    let puzzle = puzzles.slice(s![puzzle_num..puzzle_num+1, 0..IO_SIZE]).to_owned();
    let solution = solutions.slice(s![puzzle_num..puzzle_num+1, 0..IO_SIZE]).to_owned();
    let prediction = dn.predict(&puzzle);
    // println!("{}", solution);
    // println!("{}", prediction);
    let converted_solution = predict_from_one_hot(&solution);
    let converted_prediction = predict_from_one_hot(&prediction);
    //Print out the solution vs the prediction of the puzzle,
    //to see how good (or more likely bad) the net really is at Flow Free.
    println!("{}", converted_solution);
    println!("{}", converted_prediction);
    print_confidence_in_right_answer(&prediction, &solution);
}

fn predict_from_one_hot(prediction: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros((PUZZLE_WIDTH, PUZZLE_WIDTH));
    for row in 0..output.nrows() {
        for col in 0..output.ncols() {
            let mut highest_value = -1.0;
            let mut best_color = -1.0;
            for color in 0..COLORS {
                let pred = prediction[[0, row*PUZZLE_WIDTH*COLORS + col*COLORS + color]];
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
    I don't really know how to describe what this is, but I want a sequence that
    successively "scans" between 0 and 1 with increasing resolution, meaning
    it starts at 1/2, then 1/4, then 3/4, then 1/8, 3/8, 5/8, 7/8, 1/16, etc.
*/
fn interpolate_by_halves(iterations: u32) -> f32 {
    //Initialize the count and the current denominator (power of 2)
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

    //In case input is invalid (shouldn't happen with proper input)
    0.0
}

fn print_confidence_in_right_answer(prediction: &Array2<f32>, solution: &Array2<f32>) {
    let debug = false;
    //Loop over the puzzles
    for row in 0..prediction.nrows() {
        //Store the total confidence
        let mut confidence = 0.0;
        //Loop over the tiles in the puzzle, checking if each one is correct
        for col in 0..prediction.ncols() {
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
        //Take the average
        confidence /= (prediction.ncols()/COLORS) as f32;
        println!("Average confidence in correct answer: {}", confidence);
    }
}

// fn dropout(input: &mut Array2<f32>) {
//     println!("Starting dropout.");
//     let mut rng = rand::thread_rng();
//     for row in 0..input.nrows() {
//         for col in 0..input.ncols() {
//             if input[[row, col]] == 1.0 {
//                 let p = COLORS as f32 /(input.ncols() as f32);
//                 if rng.gen::<f32>() <= p {
//                     input[[row, col]] = 0.0;
//                 }
//             }
//         }
//     }
// }