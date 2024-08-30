use std::fmt::Display;
use ndarray::{Array1, Array2, Array3};
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

///The odds that a fire starts at each position (at time 0)
pub const FIRE_STARTING_CHANCE: f64 = 0.1;
///Controls how quickly the fire starting odds increase over time
///Starting at 0.0005, so after 1k it's just over 40%
pub const FIRE_EXP_PARAMETER: f32 = 0.0005;
///How long you have to react before the fire kills you
pub const FIRE_BUILDUP: u16 = 1;
///How long the fire lasts
pub const FIRE_LENGTH: u16 = 2;

///How many possible actions there are.
pub const ACTIONS_NUM: usize = 4;
///Move right.
pub const ACTION_RIGHT: u16 = 0;
///Move up.
pub const ACTION_UP: u16 = 1;
///Move left.
pub const ACTION_LEFT: u16 = 2;
///Move down.
pub const ACTION_DOWN: u16 = 3;
///Do nothing.
pub const ACTION_NOTHING: u16 = 4;

///How far the player moves per update.
pub const MOVE_SPEED: isize = 1;

///How many channels get_state will return.
pub const NUM_STATE_CHANNELS: usize = 1+FIRE_LENGTH as usize;

///An instance of the game.
#[derive(Clone)]
pub struct Environment {
    time: u128,
    alive: bool,
    position: (isize, isize),
    height: usize,
    width: usize,
    fires: Vec<Vec<u16>>,
}

impl Environment {
    pub fn new(height: usize, width: usize) -> Environment {
        Environment{
            time: 0,
            alive: true,
            position: (width as isize / 2, height as isize / 2),
            fires: vec![vec![0; width]; height],
            height,
            width
        }
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    pub fn is_player_on_fire(&self) -> bool {
        self.fires[self.position.0 as usize][self.position.1 as usize] > 0
    }

    ///Updates the time, fires, and position based on the given action
    pub fn update(&mut self, action: u16) {
        if !self.alive {
            panic!("Updated an unalive environment.");
        }

        //Update time
        self.time += 1;

        //Update fires
        let mut rng = StdRng::seed_from_u64(self.time as u64);
        for row in 0..self.height {
            for col in 0..self.width {
                if self.fires[row][col] > 0 {
                    self.fires[row][col] += 1;
                }
                else if rng.gen::<f64>() < fire_chance(self.time) {
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

        x = x.clamp(0, self.width as isize - 1);
        y = y.clamp(0, self.height as isize - 1);

        self.position = (x, y);
    }

    pub fn get_state(&self) -> Array3<f32> {
        let mut output: Array3<f32> = Array3::zeros((NUM_STATE_CHANNELS, self.height, self.width));
        let (x, y) = (self.position.0 as f32, self.position.1 as f32);
        for row in 0..self.height {
            for col in 0..self.width {
                //Player position -- each tile has value 1/(distance + 1)
                let r = row as f32;
                let c = col as f32;
                output[[0, row, col]] = 1./(((r-y)*(r-y)+(c-x)*(c-x)).sqrt() + 1.);
                //Fire states
                let fire_value = self.fires[row][col] as usize;
                output[[fire_value, row, col]] = 1.;
            }
        }

        //Player position
        //output[[1, self.position.0 as usize, self.position.1 as usize]] = 1.;

        output
    }

    pub fn time(&self) -> u128 {
        self.time
    }
}

fn fire_chance(time: u128) -> f64 {
    //FIRE_STARTING_CHANCE
    1. - ((1. - FIRE_STARTING_CHANCE) * (-FIRE_EXP_PARAMETER as f64 * time as f64).exp())
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
                    output += match (self.alive, self.is_player_on_fire()) {
                        (true, false) => " O ",
                        (false, false) => " X ",
                        (true, true) => " @ ",
                        (false, true) => " % ",
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