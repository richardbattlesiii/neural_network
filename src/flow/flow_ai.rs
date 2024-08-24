use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use ndarray::{Array1, Array2};
use rand::{random, Rng};
use rand::seq::SliceRandom;


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
        for current_value in puzzle {
            for color in 0..COLORS {
                if current_value == color as f32 {
                    one_hot_puzzle.push(1f32);
                }
                else {
                    one_hot_puzzle.push(0f32);
                }
            }
        }
        puzzles.extend(one_hot_puzzle);

        let label = chunks[row].clone();
        let mut one_hot_label = vec![];
        for current_label in label {
            for color in 0..COLORS {
                if current_label == color as f32 {
                    one_hot_label.push(1f32);
                }
                else {
                    one_hot_label.push(0f32);
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

pub fn generate_puzzles(num_puzzles: usize) -> (Array2<f32>, Array2<f32>) {
    println!("Generating puzzles...");
    let mut puzzles: Array2<f32> = Array2::zeros((0, PUZZLE_WIDTH*PUZZLE_WIDTH*COLORS));
    let mut solutions: Array2<f32> = Array2::zeros((0, PUZZLE_WIDTH*PUZZLE_WIDTH*COLORS));
    for puzzle_num in 0..num_puzzles {
        if puzzle_num % (num_puzzles / 10) == 0 {
            println!("{}%...", puzzle_num * 100 / num_puzzles);
            //println!("{}", solution);
        }
        let solution_2d = generate_solution();
        let puzzle_2d = remove_solution(&solution_2d);

        let solution_1d = one_hot_encode(&solution_2d);
        let puzzle_1d = one_hot_encode(&puzzle_2d);

        solutions.push_row(solution_1d.view()).unwrap();
        puzzles.push_row(puzzle_1d.view()).unwrap();
    }
    println!("100%");
    //println!("{}\n\n{}", puzzles.row(0), solutions.row(0));
    (puzzles, solutions)
}

pub fn generate_solution() -> Array2<f32> {
    let debug = false;
    if debug {
        println!("Generating solution.");
    }
    let mut grid: Array2<f32> = Array2::zeros((PUZZLE_WIDTH, PUZZLE_WIDTH));
    //Represents groups of tile coordinates that are still 0 and can reach each other.
    //Only includes groups with at least 2 members, so that a path can be generated.
    let mut reachable = find_reachable(&grid);
    //Start at 2 because 0 is reserved for impassable tiles and 1 is reserved for empty but passable
    let mut current_color = 2;
    while current_color < COLORS && reachable.len() > 0 {
        if debug {
            println!("Color is {}.", current_color);
        }
        let group_num: usize = (random::<f32>() * reachable.len() as f32) as usize;
        let group = &reachable[group_num];
        let group_size = group.len();
        let mut rng = rand::thread_rng();
        let mut node1 = rng.gen_range(0..group_size);
        let mut node2 = rng.gen_range(0..group_size);
        while node1 == node2 {
            node1 = rng.gen_range(0..group_size);
            node2 = rng.gen_range(0..group_size);
        }

        if debug {
            println!("Successfully generated separate nodes: {} and {}", node1, node2);
        }

        let (x1, y1) = group[node1];
        let (x2, y2) = group[node2];

        if debug {
            println!("Generating path from ({}, {}) to ({}, {}).", x1, y1, x2, y2);
            println!("Path will be in this group: {:?}", group);
        }
        let path: Vec<(usize, usize)> = vec![(x1, y1)];
        let path = generate_path(group, &path, (x1, y1), (x2, y2), (grid.nrows(), grid.ncols()));

        //Empty path means it was impossible to generate (though that shouldn't be possible...)
        if path.len() == 0 {
            panic!("Generated path was empty... how did that happen? Probably a bug in find_reachable.");
        }

        //Add the generated path to the grid.
        for node in 0..path.len() {
            let (current_x, current_y) = path[node];
            grid[[current_x, current_y]] = current_color as f32;
        }
        if debug {
            println!("Grid is now:\n{}", grid);
        }

        //Update reachable.
        reachable = find_reachable(&grid);

        current_color += 1;
    }
    if debug {
        println!("Generated solution:\n{}", grid);
    }
    grid
}

fn generate_path(
        reachable: &Vec<(usize, usize)>,
        path_so_far: &Vec<(usize, usize)>,
        (x1, y1): (usize, usize),
        (x2, y2): (usize, usize),
        max_size: (usize, usize))
        -> Vec<(usize, usize)> {

    let debug = false;
    if debug {
        println!("\tGoing from ({}, {}) to ({}, {}).", x1, y1, x2, y2);
        println!("\tI've been everywhere, man: {:?}", path_so_far);
    }
    //Get all the valid locations we can go to.
    //Array2s get printed in the format [-y, x] but I don't care.
    let mut directions = vec![];
    //Right
    if x1+1 < max_size.0{
        directions.push((x1+1, y1));
    }
    //Up
    if y1+1 < max_size.1{
        directions.push((x1, y1+1));
    }
    //Left
    if x1 > 0{
        directions.push((x1-1, y1));
    }
    //Down
    if y1 > 0{
        directions.push((x1, y1-1));
    }

    let mut i = 0;
    while i < directions.len() {
        let len = path_so_far.len();
        let mut removed = false;
        if len > 1 && path_so_far[0..len-2].contains(&directions[i]) {
            return vec![];
        }
        else if (!reachable.contains(&directions[i])) || path_so_far[path_so_far.len() - 1] == directions[i] {
            directions.remove(i);
            removed = true;
        }
        else if debug {
            println!("Direction {:?} is valid?", directions[i]);
        }
        if !removed {
            i += 1;
        }
    }

    if debug {
        println!("\tReachable: {:?}", reachable);
        println!("\tDirections: {:?}", directions);
    }
    //Check if we're adjacent to the last point.
    for i in 0..directions.len() {
        let direction = directions[i];
        if direction == (x2, y2) {
            if debug {
                println!("\tAdjacent to output node.");
            }
            let mut output = path_so_far.clone();
            output.push((x2, y2));
            if debug {
                println!("\tReturning {:?}.", output);
            }
            return output;
        }
    }

    if debug {
        println!("\tNot adjacent to output.");
    }
    //If we've already visited an adjacent tile, that means there are multiple solutions, so the
    //puzzle is invalid.
    for i in 0..directions.len() {
        let len = path_so_far.len();
        if len > 1 && path_so_far[0..len-2].contains(&directions[i]) {
            if debug {
                println!("\tpath_so_far had an adjacent node -- {:?}", path_so_far);
            }
            return vec![];
        }
    }

    //Otherwise, try going in each direction.
    directions.shuffle(&mut rand::thread_rng());
    for i in 0..directions.len() {
        let mut my_path = path_so_far.clone();
        my_path.push(directions[i].clone());
        if debug {
            println!("Trying to go to {:?}", directions[i]);
        }
        let recursive_path = generate_path(reachable, &my_path, directions[i].clone(), (x2, y2), max_size);
        if recursive_path.len() > 0 {
            if debug {
                println!("Succeeded.");
            }
            return recursive_path;
        }
        else if debug {
            println!("Failed to go to {:?}.", directions[i]);
        }
    }

    let nothing:Vec<(usize, usize)> = vec![];
    nothing
}

fn find_reachable(grid: &Array2<f32>) -> Vec<Vec<(usize, usize)>> {
    let debug = false;
    //Where we've been.
    let mut visited: Vec<(usize, usize)> = vec![];
    //Groups of connected areas with at least 2 nodes (so that a path can be made).
    let mut reachable: Vec<Vec<(usize, usize)>> = vec![];
    //Loop through every spot on the grid.
    for row in 0..grid.nrows() {
        for col in 0..grid.ncols() {
            let position = (row, col);
            if debug {
                println!("Checking ({}, {})...", row, col);
            }
            //Check if we've been here before.
            if !visited.contains(&position) {
                //If not, add it to the list.
                visited.push(position);
                //And check if this tile is empty.
                if grid[[row, col]] <= 0.01 {
                    if debug {
                        println!("Grid [{}, {}] is {}", row, col, grid[[row,col]]);
                    }
                    //If so, use the recursive function to find all connected nodes
                    let mut reached = flood_fill(grid, &position, &vec![position]);
                    reached.push(position);
                    if reached.len() > 1 {
                        visited.extend(reached.clone());
                        reachable.push(reached);
                        if debug {
                            println!("Found reachable tiles. Now reachable is:\n{:?}", reachable);
                        }
                    }
                    else if debug {
                        println!("Reached wasn't long enough: {:?}", reached);
                    }
                }
            }
        }
    }
    if debug {
        println!("Reachable tiles:\n{:?}", reachable);
    }
    reachable
}

fn flood_fill(grid: &Array2<f32>, position: &(usize, usize),
        visited: &Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let debug = false;
    let x = position.0;
    let y = position.1;
    if debug {
        println!("\tFlood fill from ({}, {})", x, y);
    }
    let mut directions = vec![];
    //Right
    if x+1 < grid.nrows() {
        directions.push((x+1, y));
    }
    //Up
    if y+1 < grid.ncols() {
        directions.push((x, y+1));
    }
    //Left
    if x > 0 {
        directions.push((x-1, y));
    }
    //Down
    if y > 0 {
        directions.push((x, y-1));
    }
    //Go in a random direction order, mostly because I can.
    directions.shuffle(&mut rand::thread_rng());

    //Keep track of where we've been but as a copy.
    let mut my_visited = visited.to_owned();
    //Keep track of the positions that the caller should add.
    let mut new_positions: Vec<(usize, usize)> = vec![];
    if debug {
        println!("\tI've been to: {:?}", my_visited);
    }
    //Go in each direction.
    for i in 0..directions.len() {
        let direction = directions[i];
        //Make sure we haven't been there and it's empty.
        if !my_visited.contains(&direction) && grid[[direction.0, direction.1]] <= 0.01 {
            new_positions.push(direction.clone());
            my_visited.push(direction.clone());
            if debug {
                println!("\tFound a new spot at ({}, {})", direction.0, direction.1);
            }
            new_positions.extend(&flood_fill(grid, &direction, &my_visited));
            my_visited.extend(new_positions.clone());
        }
    }
    new_positions
}

fn remove_solution(grid: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros((grid.nrows(), grid.ncols()));
    for row in 0..output.nrows() {
        for col in 0..output.ncols() {
            let color = grid[[row, col]];
            if color != 0.0 {
                let mut same_color_count: u8 = 0;
                for i in -1..2 as i8 {
                    for j in -1..2 as i8 {
                        if (i.abs()==1) ^ (j.abs()==1) {
                            //println!("Checking i {} and j {}", i, j);
                            if (row > 0 || i > -1) && (col > 0 || j > -1) && (row as i8 + i < grid.nrows() as i8) && (col as i8 + j < grid.ncols() as i8) {
                                if grid[[(row as i8 + i) as usize, (col as i8 + j) as usize]] == color {
                                    same_color_count += 1;
                                }
                            }
                        }
                    }
                }

                if same_color_count == 1 {
                    output[[row, col]] = color;
                    //println!("Okie");
                }
                else if same_color_count != 2 {
                    println!("Row: {}, Col: {}", row, col);
                    panic!("Had {} same_color_count.", same_color_count);
                }
            }
        }
    }

    output
}

fn one_hot_encode(grid: &Array2<f32>) -> Array1<f32> {
    let mut output = Array1::zeros(COLORS * PUZZLE_WIDTH * PUZZLE_WIDTH);

    for row in 0..PUZZLE_WIDTH {
        for col in 0..PUZZLE_WIDTH {
            let grid_value = grid[[row, col]];
            output[[row*COLORS*PUZZLE_WIDTH + col*COLORS + (grid_value as usize)]] = 1.0
        }
    }

    output
}