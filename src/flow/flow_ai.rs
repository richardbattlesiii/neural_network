use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use crate::matrix::*;

pub const PUZZLE_WIDTH: usize = 16;
pub const COLORS: usize = 80;
pub const KEY: &str = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()=+;':\"[]\\{}|";
pub fn convert() -> io::Result<(Matrix, Matrix)> {
    // Open the file in read-only mode
    let mut file = File::open("512 16x16.txt")?;

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


    // Create Vec<Vec<i32>> with dimensions [filelength/interval x interval]
    let chunks: Vec<Vec<f32>> = values.chunks(PUZZLE_WIDTH*PUZZLE_WIDTH)
        .map(|chunk| chunk.to_vec())
        .collect();

    let mut puzzles: Vec<f32> = vec![];
    let mut solutions: Vec<f32> = vec![];
    for row in (0..chunks.len()-1).step_by(2) {
        let puzzle = chunks[row+1].clone();
        puzzles.extend(puzzle);

        let label = chunks[row].clone();
        solutions.extend(label);
    }

    //println!("Chunks:\n{}", chunks.len());
    let puzzle_matrix = Matrix {
        values: puzzles,
        rows: values.len()/(2*PUZZLE_WIDTH*PUZZLE_WIDTH),
        cols: PUZZLE_WIDTH*PUZZLE_WIDTH
    };

    let solution_matrix = Matrix {
        values: solutions,
        rows: values.len()/(2*PUZZLE_WIDTH*PUZZLE_WIDTH),
        cols: PUZZLE_WIDTH*PUZZLE_WIDTH
    };
    //println!("{}", solution_matrix);
    Ok((puzzle_matrix, solution_matrix))
}

