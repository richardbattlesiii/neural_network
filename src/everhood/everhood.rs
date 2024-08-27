use std::fmt::Display;
use ndarray::{Array1, Array2};
use rand::random;

///The odds that a fire starts at each position (at time 0)
pub const FIRE_STARTING_CHANCE: f32 = 0.05;
///Controls how quickly the fire starting odds increase over time
///Starting at 0.0005, so after 1k it's just over 40%
pub const FIRE_EXP_PARAMETER: f32 = 0.0005;
///How long you have to react before the fire kills you
pub const FIRE_BUILDUP: u8 = 3;
///How long the fire lasts
pub const FIRE_LENGTH: u8 = 4;

///How many possible actions there are.
pub const ACTIONS_NUM: u8 = 5;
///Do nothing.
pub const ACTION_NOTHING: u8 = 0;
///Move right.
pub const ACTION_RIGHT: u8 = 1;
///Move up.
pub const ACTION_UP: u8 = 2;
///Move left.
pub const ACTION_LEFT: u8 = 3;
///Move down.
pub const ACTION_DOWN: u8 = 4;

///How far the player moves per update.
pub const MOVE_SPEED: f32 = 0.5;

///An instance of the game.
pub struct Environment {
    time: u128,
    alive: bool,
    position: (f32, f32),
    height: usize,
    width: usize,
    fires: Vec<Vec<u8>>,
}

impl Environment {
    pub fn new(height: usize, width: usize) -> Environment {
        Environment{
            time: 0,
            alive: true,
            position: (width as f32 / 2., height as f32 / 2.),
            fires: vec![vec![0; width]; height],
            height,
            width
        }
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    ///Updates the time, fires, and position based on the given action
    pub fn update(&mut self, action: u8) {
        if !self.alive {
            panic!("Updated an unalive environment.");
        }

        //Update time
        self.time += 1;

        //Update fires
        for row in 0..self.height {
            for col in 0..self.width {
                if self.fires[row][col] > 0 {
                    self.fires[row][col] += 1;
                }
                else if random::<f64>() < fire_chance(self.time) {
                    self.fires[row][col] = 1;
                }
                if self.fires[row][col] > FIRE_LENGTH {
                    self.fires[row][col] = 0;
                }
            }
        }

        let (mut x, mut y) = self.position;

        let rounded_x = x as usize;
        let rounded_y = y as usize;

        if self.fires[rounded_y][rounded_x] > FIRE_BUILDUP {
            self.alive = false;
        }
        else {
            match action {
                ACTION_NOTHING => {}

                ACTION_RIGHT => {x += MOVE_SPEED}
                ACTION_UP =>    {y += MOVE_SPEED}
                ACTION_LEFT =>  {x -= MOVE_SPEED}
                ACTION_DOWN =>  {y -= MOVE_SPEED}
                
                invalid_action => panic!("Invalid action taken: {}", invalid_action)
            }
        }

        x = x.clamp(0., (self.width-1) as f32);
        y = y.clamp(0., (self.height-1) as f32);

        self.position = (x, y);
    }

    pub fn get_state(&self) -> Array1<f32> {
        let len = 4 + self.height*self.width;
        let mut output: Array1<f32> = Array1::zeros(len);

        output[[0]] = self.time as f32;
        output[[1]] = self.alive as u8 as f32;
        output[[2]] = self.position.0;
        output[[3]] = self.position.1;

        for row in 0..self.height {
            for col in 0..self.width {
                output[[4 + row*self.width + col]] = self.fires[row][col] as f32;
            }
        }
        /*
            time: u128,
            alive: bool,
            position: (f32, f32),
            height: usize,
            width: usize,
            fires: Vec<Vec<u8>>,
        */
        output
    }
}

impl Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        let (x, y) = self.position;
        let player_x = x as usize;
        let player_y = y as usize;
        for row in 0..self.height {
            for col in 0..self.width {
                if col == player_x && row == player_y {
                    if self.alive {
                        output += " O ";
                    }
                    else {
                        output += " X ";
                    }
                }
                else {
                    let fire_value = self.fires[row][col];
                    if fire_value == 0 {
                        output += " . ";
                    }
                    else {
                        output += &format!(" {} ", self.fires[row][col]);
                    }
                }
            }
            output += "\n";
        }
        output += "\n";
        write!(f, "{}", output)
    }
}

fn fire_chance(time: u128) -> f64 {
    return 1. - (1. - FIRE_STARTING_CHANCE) as f64 * (-FIRE_EXP_PARAMETER as f64 * time as f64).exp()
}