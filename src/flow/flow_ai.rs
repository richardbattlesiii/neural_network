use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use ndarray::{Array, Array2};

pub const PUZZLE_WIDTH: usize = 4;
pub const COLORS: usize = 10;
pub const KEY: &str = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()=+;':\"[]\\{}|";
pub fn convert() -> io::Result<(Array2<f32>, Array2<f32>)> {
    // Open the file in read-only mode
    let mut file = File::open("testing.txt")?;

    // Create a String to hold the file contents
    let mut contents = String::new();

    // Read the file contents into the String
    file.read_to_string(&mut contents)?;

    let mut key_map = HashMap::new();

    for (i, c) in KEY.chars().enumerate() {
        key_map.insert(c, (i as i32) as f32);
    }
    
    let values: Vec<f32> = contents
        .chars()
        .filter_map(|c| key_map.get(&c).cloned())
        .collect();


    let chunks: Vec<Vec<f32>> = values.chunks(PUZZLE_WIDTH*PUZZLE_WIDTH)
        .map(|chunk| chunk.to_vec())
        .collect();

    let mut puzzles: Vec<f32> = vec![];
    let mut solutions: Vec<f32> = vec![];
    for row in (0..chunks.len()).step_by(2) {
        let puzzle = chunks[row+1].clone();
        let mut one_hot_puzzle = vec![];
        for tile in 0..puzzle.len() {
            for color in 0..COLORS {
                if puzzle[tile] == color as f32 {
                    one_hot_puzzle.push(1.0);
                }
                else {
                    one_hot_puzzle.push(0.0);
                }
            }
        }
        puzzles.extend(one_hot_puzzle);

        let label = chunks[row].clone();
        let mut one_hot_label = vec![];
        for tile in 0..label.len() {
            for color in 0..COLORS {
                if label[tile] == color as f32 {
                    one_hot_label.push(1.0);
                }
                else {
                    one_hot_label.push(0.0);
                }
            }
        }
        solutions.extend(one_hot_label);
    }

    let rows = values.len()/(2*PUZZLE_WIDTH*PUZZLE_WIDTH);
    let cols = COLORS*PUZZLE_WIDTH*PUZZLE_WIDTH;
    println!("Length: {}, Rows: {}, Cols: {}", puzzles.len(), rows, cols);
    let puzzles_output = Array2::from_shape_vec((rows, cols), puzzles).unwrap();
    let solutions_output = Array2::from_shape_vec((rows, cols), solutions).unwrap();

    Ok((puzzles_output, solutions_output))
}

