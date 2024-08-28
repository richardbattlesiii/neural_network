use rand::Rng;
use scrap::{Capturer, Display};
use std::io::ErrorKind::WouldBlock;
use std::fs::File;
use std::thread;
use std::time::Duration;

use crate::everhood::everhood;
use crate::everhood::everhood::Environment;
use crate::helpers::activation_functions::RELU;
use crate::layers::convolutional_layer::CONVOLUTION_BASIC;
use super::convolutional_net::ConvolutionalNet;

pub fn make_gamer_net() {
    let width = 15;
    let height = 15;
    let max_episodes = 10;
    let mut episode = 0;
    let mut rng = rand::thread_rng();
    let mut cn = ConvolutionalNet::new(
        width,
        1,
        &[32, 64],
        &[3, 5],
        &[5*width*height, everhood::ACTIONS_NUM as usize],
        &[RELU, RELU, RELU, RELU, RELU, RELU, RELU, RELU],
        CONVOLUTION_BASIC,
    );
    while episode < max_episodes {
        let mut env = Environment::new(width, height);
        let q_values = cn.predict(&env.get_state().view());
        while env.is_alive() {
            env.update(rng.gen_range(0..everhood::ACTIONS_NUM));
        }
        println!("{}", env);
        episode += 1;
    }
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