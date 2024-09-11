use ndarray::prelude::*;

///The ndarray crate doesn't have an inverse function, and the ndarray-linalg crate didn't want to work :)
pub fn invert(input: &Array2<f32>) -> Array2<f32> {
    let debug = false;
    let mut augmented: Array2<f32> = input.to_owned();
    augmented.append(Axis(1), Array2::eye(input.nrows()).view()).unwrap();
    for row in 0..input.nrows() {
        if debug {
            println!("{}\n", augmented);
        }
        //Find a row where the column is non-zero
        if augmented[[row, row]] == 0. {
            for check_row in row+1..input.nrows() {
                if augmented[[check_row, row]] != 0. {
                    swap_rows(&mut augmented, row, check_row);
                    break;
                }
            }
            if augmented[[row, row]] == 0. {
                panic!("Couldn't find a non-zero value in col {row}.");
            }
        }

        //Normalize
        let value = augmented[[row, row]];
        let mut current_row = augmented.row_mut(row);
        current_row /= value;
        let current_row = current_row.to_owned();

        //Eliminate below
        for eliminate_row in row+1..input.nrows() {
            let coefficient = augmented[[eliminate_row, row]];
            let mut eliminate = augmented.row_mut(eliminate_row);
            eliminate.scaled_add(-coefficient, &current_row);
        }
    }

    //And now go back up
    for source_row in (0..input.nrows()).rev() {
        let current_row = augmented.row(source_row).to_owned();
        if debug {
            println!("{}", augmented);
        }
        //Eliminate above
        for eliminate_row in 0..source_row {
            let coefficient = augmented[[eliminate_row, source_row]];
            let mut eliminate = augmented.row_mut(eliminate_row);
            eliminate.scaled_add(-coefficient, &current_row);
        }
    }

    if debug {
        println!("End:\n{}\n", augmented);
    }
    augmented.slice(s![.., input.ncols()..]).to_owned()
}

pub fn swap_rows(input: &mut Array2<f32>, row1: usize, row2: usize) {
    for col in 0..input.ncols() {
        input.swap([row1, col], [row2, col]);
    }
}