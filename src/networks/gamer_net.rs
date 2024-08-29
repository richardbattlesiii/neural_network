use rand::{thread_rng, Rng};
use scrap::{Capturer, Display};
use std::io::ErrorKind::WouldBlock;
use std::fs::File;
use std::thread;
use std::time::{Duration, Instant};
use ndarray::{ArrayD, Axis};
use crate::everhood::everhood::{self, ACTIONS_NUM};
use crate::everhood::everhood::Environment;
use crate::helpers::activation_functions::{RELU, SIGMOID};
use crate::{convolutional_layer, prelude::*};
use super::neural_net::NeuralNet;

const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 0.01;
const LAMBDA: f32 = 0.5;

const MAX_EPISODES: i32 = 10000;
const PRINTERVAL: i32 = 50;
const ENV_PRINTERVAL: i32 = PRINTERVAL*10;
const HEIGHT: usize = 5;
const WIDTH: usize = 5;

pub fn make_gamer_net() {
    let mut episode = 1;

    let mut net = NeuralNet::new();

    //Convolutional layers
    let size = everhood::NUM_STATE_CHANNELS*HEIGHT*WIDTH;
    let num_channels = &[everhood::NUM_STATE_CHANNELS, 8, 16];
    let convolutional_layer_sizes = &[3, 3, 5];
    for i in 0..num_channels.len()-1 {
        net.add_layer(Box::from(ConvolutionalLayer::new(
                HEIGHT,
                num_channels[i],
                num_channels[i+1],
                convolutional_layer_sizes[i],
                LEARNING_RATE,
                LAMBDA,
                RELU,
                convolutional_layer::CONVOLUTION_BASIC)));
    }

    //Reshaping layer
    let last_convolutional_layer_shape = vec![num_channels[num_channels.len()-1], HEIGHT, WIDTH];
    let dense_layer_input_shape = num_channels[num_channels.len()-1]*HEIGHT*WIDTH;
    net.add_layer(Box::from(ReshapingLayer::new(
        last_convolutional_layer_shape,
        vec![dense_layer_input_shape]
    )));

    //Dense layers
    let dense_layer_sizes = &[dense_layer_input_shape, everhood::ACTIONS_NUM];
    let num_dense_layers = dense_layer_sizes.len() - 1;
    for i in 0..dense_layer_sizes.len()-2 {
        net.add_layer(Box::from(DenseLayer::new(
            dense_layer_sizes[i],
            dense_layer_sizes[i+1],
            LEARNING_RATE,
            LAMBDA,
            RELU
        )));
    }

    //Outputlayer
    net.add_layer(Box::from(DenseLayer::new(
        dense_layer_sizes[num_dense_layers-1],
        dense_layer_sizes[num_dense_layers],
        LEARNING_RATE,
        LAMBDA,
        SIGMOID
    )));
    // net.add_layer(Box::from(SoftmaxLayer::new(
    //     everhood::ACTIONS_NUM,
    //     1,
    // )));

    net.initialize();

    //Start the actual training
    let mut epsilon = 0.4;
    let start = Instant::now();
    while episode < MAX_EPISODES {
        let mut env = Environment::new(HEIGHT, WIDTH);
        while env.is_alive() {
            if episode % (ENV_PRINTERVAL) == 0 {
                println!("{}", env);
            }
            let current_state = env.get_state().into_dyn();
            let converted_state = current_state.view().insert_axis(Axis(0));
            let q_values = net.predict(&converted_state);
            if q_values.is_any_nan() {
                //println!("Env:\n{}", env);
                println!("Q values:\n{}\n", q_values);
                println!("Time: {}, Episode: {}\n", env.time(), episode);
                panic!("NANIC PANIC\n");
            }
            let action = get_action(&q_values, epsilon);
            env.update(action.0);
            let reward;
            let max_predicted_q;
            if env.is_alive() {
                reward = (env.time() as f32).sqrt()/4.;
                max_predicted_q = max_predicted_next_q(&net, &env);
            }
            else {
                reward = -1.;
                max_predicted_q = 0.;
            }
            //println!("Reward at time {} is {}.", env.time(), reward);
            let future_value = reward + GAMMA*max_predicted_q - &q_values;
            let mut labels: ArrayD<f32> = q_values.clone();
            labels.scaled_add(LEARNING_RATE, &future_value);
            net.backpropagate(&converted_state, &labels.view());
        }
        if episode % (ENV_PRINTERVAL) == 0 {
            println!("{}", env);
        }
        if episode % PRINTERVAL == 0 {
            println!("Lasted for {} steps on episode {}.", env.time(), episode);
            //println!("Epsilon: {}", epsilon);
        }
        epsilon *= 0.9999;
        episode += 1;
    }
    let duration = start.elapsed().as_millis();
    println!("Finished in {}ms.", duration);
}

fn get_action(q_values: &ArrayD<f32>, epsilon: f32) -> (u16, f32) {
    //println!("Getting action from:\n{}", q_values);
    let q_values = q_values.to_shape(q_values.len()).unwrap();
    let mut rng = rand::thread_rng();

    //Explore
    if rng.gen::<f32>() < epsilon {
        let random_index = rng.gen_range(0..q_values.len()) as u16;
        (random_index, q_values[[random_index as usize]])
    }
    else {
        //Exploit
        let mut max = -f32::MAX;
        let mut max_index = u16::MAX;
        for i in 0..q_values.len() {
            if q_values[[i]] > max {
                max = q_values[[i]];
                max_index = i as u16;
            }
        }
        (max_index, max)
    }
}

fn max_predicted_next_q(net: &NeuralNet, env: &Environment) -> f32 {
    if !env.is_alive() {
        return 0.0;
    }
    let mut max = f32::MIN;
    for action in 0..ACTIONS_NUM {
        let mut env = env.clone();
        env.update(action as u16);
        let current_q_values = net.predict(&env.get_state().into_dyn().view().insert_axis(Axis(0)));
        let current_best = get_action(&current_q_values, 0.).1;
        if current_best > max {
            max = current_best;
        }
    }
    max
}










pub fn test_screenshot() {
    let one_second = Duration::new(1, 0);
    let one_frame = one_second / 60;

    let display = Display::primary().expect("Couldn't find primary display.");
    let mut capturer = Capturer::new(display).expect("Couldn't begin capture.");
    let (w, h) = (capturer.width(), capturer.height());

    loop {
        // Wait until there's a frame.

        let buffer = match capturer.frame() {
            Ok(buffer) => buffer,
            Err(error) => {
                if error.kind() == WouldBlock {
                    // Keep spinning.
                    thread::sleep(one_frame);
                    continue;
                } else {
                    panic!("Error: {}", error);
                }
            }
        };

        println!("Captured! Saving...");

        // Flip the ARGB image into a BGRA image.

        let mut bitflipped = Vec::with_capacity(w * h);
        let stride = buffer.len() / h;
        let step_by_amount = 16;
        for y in (0..h).step_by(step_by_amount) {
            for x in (0..w).step_by(step_by_amount) {
                let i = stride * y + 4 * x;
                bitflipped.extend_from_slice(&[
                    buffer[i + 2],
                    buffer[i + 1],
                    buffer[i],
                    255,
                ]);
            }
        }

        // Save the image.

        repng::encode(
            File::create("screenshot.png").unwrap(),
            (w/step_by_amount) as u32,
            (h/step_by_amount) as u32,
            &bitflipped,
        ).unwrap();

        println!("W: {}, H: {}", w/step_by_amount, h/step_by_amount);
        println!("Image saved to `screenshot.png`.");
        break;
    }
}