use ndarray::Array2;

pub fn pass(input: &Array2<f32>, puzzle_width: usize) -> Array2<f32> {
    let num_tiles = puzzle_width * puzzle_width;
    let num_colors = input.ncols() / num_tiles;
    
    // Create an output array with the same shape as the input
    let mut output = Array2::zeros(input.raw_dim());

    // Iterate over each batch
    for batch in 0..input.nrows() {
        // Iterate over each tile
        for tile in 0..num_tiles {
            // Calculate the start and end indices for the current tile's color probabilities
            let start = tile * num_colors;
            let end = start + num_colors;

            // Find the max value for numerical stability
            let mut max = f32::NEG_INFINITY;
            for i in start..end {
                if input[[batch, i]] > max {
                    max = input[[batch, i]];
                }
            }

            // Compute the exponentials and sum
            let mut sum = 0.0;
            let mut exp_values = vec![0.0; num_colors];
            for i in start..end {
                let exp_value = (input[[batch, i]] - max).exp();
                exp_values[i - start] = exp_value;
                sum += exp_value;
            }

            // Normalize to get softmax values
            for i in start..end {
                output[[batch, i]] = exp_values[i - start] / sum;
            }
        }
    }

    output
}
