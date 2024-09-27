use core::f32;

use crate::environments::environment::Environment;
use ndarray::Array3;
use rand::random;

pub struct CartPole {
    position: f32,
    bounds: f32, //simulation ends when exiting +- this bound

    velocity: f32,
    max_velocity: f32, //clamped
    
    pole_angle: f32,
    minimum_cosine: f32, //simulation ends when [cos(angle) < this value]

    pole_angular_velocity: f32,
    max_angular_velocity: f32, //clamped
    
    gravity: f32,
    
    mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    
    length: f32,  //half the pole's length, for math reasons

    force_magnitude: f32,

    time_step: f32,
    time: f32,
    max_time: f32, //simulation ends (successfully) when over this amount of time
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl CartPole {
    pub fn new() -> Self {
        CartPole {
            position: 0.0,
            bounds: 2.5,

            velocity: 0.0,
            max_velocity: 10.0,

            pole_angle: 0.0,
            minimum_cosine: 0.8,

            pole_angular_velocity: random::<f32>()*0.2 - 0.1,
            max_angular_velocity: 5.0,

            gravity: 10.0,

            mass_cart: 1.0,
            mass_pole: 0.1,
            total_mass: 1.1,    //mass_cart + mass_pole
            
            length: 0.5,        //actual length of pole is 1
            
            force_magnitude: 15.0,
            time_step: 0.02,    //time step
            time: 0.0,
            max_time: 10.0,
        }
    }
}

impl Environment for CartPole {
    ///Uses formulas I found online to apply a force to the cart and update everything as a result.
    fn update(&mut self, action: u16) {
        let force = match action {
            //Left
            0 => -self.force_magnitude,
            //Right
            1 => self.force_magnitude,
            //Stay still (not always used)
            _ => 0.0,
        };

        let cos_theta = self.pole_angle.cos();
        let sin_theta = self.pole_angle.sin();

        //silly little intermediate value used in both the angular and linear acceleration formulas
        let force_acceleration = (force + self.mass_pole * self.length * self.pole_angular_velocity.powi(2) * sin_theta) / self.total_mass;
        
        let theta_acc = (self.gravity * sin_theta - cos_theta * force_acceleration)
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta.powi(2) / self.total_mass));
        let x_acc = force_acceleration - self.mass_pole * self.length * theta_acc * cos_theta / self.total_mass;

        //update position
        self.position += self.time_step * self.velocity;

        //update velocity
        self.velocity += self.time_step * x_acc;
        //clamp velocity
        if self.velocity.abs() > self.max_velocity {
            self.velocity = self.velocity.clamp(-self.max_velocity, self.max_velocity);
        }

        //update angle
        self.pole_angle += self.time_step * self.pole_angular_velocity;
        //set angle from -pi to pi
        self.pole_angle = (self.pole_angle.sin()).atan2(self.pole_angle.cos());
        
        //update angular velocity
        self.pole_angular_velocity += self.time_step * theta_acc;
        //clamp angular velocity
        if self.pole_angular_velocity.abs() > self.max_angular_velocity {
            self.pole_angular_velocity = self.pole_angular_velocity.clamp(-self.max_angular_velocity, self.max_angular_velocity);
        }

        //update time
        self.time += self.time_step;
    }

    ///Returns the position and velocity of the cart and pole
    fn get_state(&self) -> Array3<f32> {
        Array3::from_shape_vec((1, 1, 4), vec![
            self.position,
            self.velocity,
            self.pole_angle,
            self.pole_angular_velocity,
        ]).unwrap()
    }

    ///Returns a weighted average of the cosine of the pole angle and distance from the center,
    ///normalized between 0 and 1.  
    fn reward(&self) -> f32 {
        if self.is_done() {
            return 0.0;
        }
        //Reward scales continuously from 0 to 1 using the cosine of the angle
        //(cosine because 0 angle represents "up")
        let angle_penalty = (self.pole_angle.cos() - self.minimum_cosine)  / (1. - self.minimum_cosine);
        let angle_weight = 0.7;

        //Higher reward the closer the cart is to position 0
        let position_reward = (self.bounds - self.position.abs()) / self.bounds;
        let position_weight = 1. - angle_weight;

        let total_reward = angle_weight*angle_penalty + position_weight*position_reward;
        total_reward*total_reward*total_reward
    }
    
    ///Left, Right = 2
    fn num_actions(&self) -> usize {
        2
    }
    
    ///Returns true when: A. the pole falls too far, B. the cart moves too far,
    ///or C. the time limit is reached.
    fn is_done(&self) -> bool {
        let angle_out_of_bounds = self.pole_angle.cos() < self.minimum_cosine;
        let out_of_touch = self.bounds < self.position || self.position < -self.bounds;
        let out_of_time = self.time > self.max_time;

        angle_out_of_bounds || out_of_touch || out_of_time
    }
}