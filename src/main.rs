#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![feature(portable_simd)]
pub mod helpers;
pub mod layers;
pub mod flow;
pub mod networks;

use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use rand::random;
use std::f32::consts::PI;
use std::time::{Instant, Duration};
use flow::flow_ai::*;
use helpers::{activation_functions, gpu_matrix, matrix};
use layers::dense_layer::DenseLayer;
use networks::dense_net::DenseNet;
use matrix::*;

const NEGATIVE_ONE_WRONG_PENALTY: f32 = 2.0;
const EXPONENTIAL_PARAMETER: f32 = -0.01;
const OSCILLATION_PARAMETER: f32 = PI/50.0;
const OSCILLATION_COEFFICIENT: f32 = 0.5;
const OSCILLATE_FOREVER: bool = true;

static NUM_THREADS:u32 = 12;
static NUM_TRIES:u32 = 1000;
static THREAD_ITERATION_NOISE_RANGE:f32 = 0.2;
static EPOCH_INCREASE:u32 = 1;

static HIGH_LEARNING_RATE:f32 = 0.4;
static LOW_LEARNING_RATE:f32 = 0.1;
static MAX_EPOCHS:u32 = 100;
static NUM_TESTING_PUZZLES:usize = 128;
static NUM_TRAINING_PUZZLES:usize = 512;
static PRINTERVAL:u32 = 1;

const IO_SIZE:usize = SIZE*SIZE;
const NUMBER_OF_LAYERS:usize = 5;
const LAYER_SIZES: [usize; NUMBER_OF_LAYERS] = [IO_SIZE, 8, 8, 8, IO_SIZE];
const ACTIVATION_FUNCTIONS: [u8; NUMBER_OF_LAYERS-1] = [0, 0, 0, 0];

fn main() {
    // if std::is_x86_feature_detected!("avx512fp16") {
    //     println!("Supported!");
    // }
    // else {
    //     println!("Not supported.");
    // }
    //make_regular_dense_net();
    genetic_algorithm();
}

fn make_regular_dense_net() {
    let (puzzles, solutions) = convert().unwrap();

    let training_puzzles = puzzles.sub_matrix(0, 0, NUM_TRAINING_PUZZLES, IO_SIZE);
    let training_solutions = solutions.sub_matrix(0, 0, NUM_TRAINING_PUZZLES, IO_SIZE);

    let testing_puzzles = puzzles.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);
    let testing_solutions = solutions.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);

    let mut dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
    dn.initialize();
    let start = Instant::now();
    
    for epoch in 0..MAX_EPOCHS {
        if epoch % PRINTERVAL == 0 {
            println!("Epoch: {}, Loss: {}", epoch, DenseNet::calculate_mse_loss(&dn.predict(&testing_puzzles), &testing_solutions));
        }
        dn.batch_backpropagate(&training_puzzles, &training_solutions);
        dn.rose_decay(epoch, LOW_LEARNING_RATE, HIGH_LEARNING_RATE, OSCILLATE_FOREVER, OSCILLATION_COEFFICIENT, OSCILLATION_PARAMETER, EXPONENTIAL_PARAMETER);
    }

    let duration = start.elapsed().as_millis();
    println!("Finished in {}ms.", duration);
}

fn genetic_algorithm() {

    let (puzzles, solutions) = convert().unwrap();
    
    let puzzle_tuple_original = Arc::new(Mutex::new((puzzles, solutions)));

    let worst = DenseNet::default();
    let best = Arc::new(Mutex::new((worst, f32::MAX)));
    let mut current_best = f32::MAX;
    for i in 0..NUM_TRIES {
        // Create a vector to hold the handles of the spawned threads
        let mut handles = vec![];

        for thread_num in 0..NUM_THREADS {
            let best = Arc::clone(&best);
            let puzzle_tuple = Arc::clone(&puzzle_tuple_original);
            
            // Spawn a new thread and train a neural net
            let handle = thread::spawn(move || {
                if thread_num == 0 {
                    println!("Starting attempt {}...", i+1);
                }
                let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();

                let puzzles = puzzle_tuple_thread.0.sub_matrix(0, 0, NUM_TRAINING_PUZZLES, IO_SIZE);
                let solutions = puzzle_tuple_thread.1.sub_matrix(0, 0, NUM_TRAINING_PUZZLES, IO_SIZE);
                drop(puzzle_tuple_thread);

                let mut dn;
                if i == 0 {
                    dn = DenseNet::new_with_arrays(&LAYER_SIZES, &ACTIVATION_FUNCTIONS);
                    dn.initialize();
                }
                else {
                    let best_thread = best.lock().unwrap();
                    dn = best_thread.0.clone();
                    if thread_num != 0 {
                        dn.add_noise(THREAD_ITERATION_NOISE_RANGE);
                    }
                }

                let mut result = -1.0;
                for epoch in 0..MAX_EPOCHS+i*EPOCH_INCREASE {
                    if thread_num == 0 && epoch % ((MAX_EPOCHS+EPOCH_INCREASE*i) / 5) == 0 {
                        println!(" {}%...", epoch*100/(MAX_EPOCHS+EPOCH_INCREASE*i));
                    }
                    result = dn.batch_backpropagate(&puzzles, &solutions);
                }

                let puzzle_tuple_thread = puzzle_tuple.lock().unwrap();
                let test_puzzles = puzzle_tuple_thread.0.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);
                let test_solutions = puzzle_tuple_thread.1.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);
                drop(puzzle_tuple_thread);
                
                let mut final_loss = 0.0;
                for test_puzzle_num in 0..NUM_TESTING_PUZZLES {
                    let test_puzzle = test_puzzles.sub_matrix(test_puzzle_num, 0, 1, IO_SIZE);
                    let test_prediction = dn.predict(&test_puzzle);
                    let test_solution = test_solutions.sub_matrix(test_puzzle_num, 0, 1, IO_SIZE);
                    let loss = DenseNet::calculate_mse_loss(&test_prediction, &test_solution);
                    final_loss += loss;
                }
                final_loss /= NUM_TESTING_PUZZLES as f32;
                let mut best_thread = best.lock().unwrap();
                if best_thread.1 > final_loss {
                    best_thread.0 = dn;
                    best_thread.1 = final_loss;
                    println!("New best from thread {} -- {}, training loss: {}", thread_num, final_loss, result);
                }
                drop(best_thread);
                
            });

            // Store the handle in the vector
            handles.push(handle);
        }

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }

        let best_main = best.lock().unwrap();
        if best_main.1 < current_best {
            current_best = best_main.1;
            let puzzle_tuple = puzzle_tuple_original.lock().unwrap();
            let randy:f32 = random();
            let puzzle_num = (randy*(puzzle_tuple.0.rows as f32)) as usize;
            let input = puzzle_tuple.0.sub_matrix(puzzle_num, 0, 1, IO_SIZE);
            let mut solution = puzzle_tuple.1.sub_matrix(puzzle_num, 0, 1, IO_SIZE);
            let mut prediction = best_main.0.predict(&input);
            solution.rows = SIZE;
            solution.cols = SIZE;
            prediction.rows = SIZE;
            prediction.cols = SIZE;

            println!("\n{}\n{}", solution, prediction);
            let mut output_file = File::create("net.txt").unwrap();
            output_file.write_all(best_main.0.write_net_params_to_string().as_bytes()).unwrap();
        }
        drop(best_main);
    }

    let best = &best.lock().unwrap().0;
    let puzzle_tuple_thread = puzzle_tuple_original.lock().unwrap();
    let test_puzzles = puzzle_tuple_thread.0.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);
    let test_solutions = puzzle_tuple_thread.1.sub_matrix(NUM_TRAINING_PUZZLES, 0, NUM_TESTING_PUZZLES, IO_SIZE);
    drop(puzzle_tuple_thread);
    
    let mut final_loss = 0.0;
    for test_puzzle_num in 0..NUM_TESTING_PUZZLES {
        let test_puzzle = test_puzzles.sub_matrix(test_puzzle_num, 0, 1, IO_SIZE);
        let test_prediction = best.predict(&test_puzzle);
        let test_solution = test_solutions.sub_matrix(test_puzzle_num, 0, 1, IO_SIZE);
        let loss = DenseNet::calculate_mse_loss(&test_prediction, &test_solution);
        final_loss += loss;
    }
    final_loss /= NUM_TESTING_PUZZLES as f32;
    println!("Final results: {}", final_loss);

    println!("All done.");

    // // let a = matrix::rand(4096, 4096, 1.0);
    // // let b = matrix::rand(4096, 4096, 1.0);
    // println!("Starting...");
    // // let start = Instant::now();
    // // Matrix::merge(&a, &a, &a, &a);

    // // let duration = start.elapsed();
    // // println!("Time elapsed: {} ms", duration.as_millis());
    // let printerval = 1;
    
    // let mut dn = DenseNet::new(layer_sizes, activation_functions);
    // dn.initialize();

    // // print_results(&layers, &inputs);
    // let start = Instant::now();
    // let mut best_loss = f32::MAX;
    // for epoch in 0..max_epochs {
    //     dn.rose_decay(epoch, low_learning_rate, high_learning_rate, OSCILLATE_FOREVER,
    //         OSCILLATION_COEFFICIENT, OSCILLATION_PARAMETER, EXPONENTIAL_PARAMETER);
    //     let mut average_loss = 0.0;
    //     for i in puzzles.rows-num_testing_puzzles..puzzles.rows-1 {
    //         let input = puzzles.sub_matrix(i, 0, 1, io_size);
    //         let solution = solutions.sub_matrix(i, 0, 1, io_size);
    //         let prediction = dn.predict(input);
    //         average_loss += DenseNet::calculate_mse_loss(&prediction, &solution);
    //     }
    //     average_loss /= (num_testing_puzzles) as f32;
    //     if epoch % printerval == 0 {
    //         println!("Epoch: {}\tAverage loss: {}\tLearning rate: {}", epoch, average_loss, dn.get_learning_rate());
    //     }
    //     if average_loss < best_loss*0.9 {
    //         best_loss = average_loss;
    //         if epoch >= printerval {
                // let randy:f32 = random();
                // let puzzle_num = (randy*(puzzles.rows as f32)) as usize;
                // let input = puzzles.sub_matrix(puzzle_num, 0, 1, io_size);
                // let mut solution = solutions.sub_matrix(puzzle_num, 0, 1, io_size);
                // let mut prediction = dn.predict(input);
                // solution.rows = SIZE;
                // solution.cols = SIZE;
                // prediction.rows = SIZE;
                // prediction.cols = SIZE;

                // println!("\nNew best: {}\n{}\n{}", average_loss, solution, prediction);
    //         }
    //     }
    //     let mut training_puzzles = puzzles.sub_matrix(0, 0, num_training_puzzles, io_size);
    //     training_puzzles.add(&rand(num_training_puzzles, io_size, 0.3));
    //     let training_solutions = solutions.sub_matrix(0, 0, num_training_puzzles, io_size);
    //     dn.batch_backpropagate(&training_puzzles, &training_solutions);
    // }
    // let duration = start.elapsed();
    // println!("Time elapsed: {} ms", duration.as_millis());
    // print_results(&layers, &inputs);
}





fn print_results(dn: DenseNet, inputs: &Matrix) {
    for i in 0..inputs.rows {
        let input = inputs.sub_matrix(i, 0, 1, inputs.cols);
        println!("Input & prediction:\n{}", input);
        println!("{}", dn.predict(&input));
    }
    println!();
}

