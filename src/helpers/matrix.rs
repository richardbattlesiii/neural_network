use std::fmt;
use rand::Rng;

const STRASSEN_CUTOFF:usize = 64;
//cost SIMD_LANES:usize = 64;

#[derive(Default)]
#[derive(Debug)]
pub struct Matrix {
    pub values: Vec<f32>,
    pub rows: usize,
    pub cols: usize
}

pub fn rand(rows:usize, cols:usize, max_value: f32) -> Matrix {
    let mut rng = rand::thread_rng();
    let mut output_values = Vec::with_capacity(rows*cols);
    for _ in 0..rows {
        for _ in 0..cols {
            output_values.push(rng.gen::<f32>()*max_value);
        }
    }
    Matrix {
        values: output_values,
        rows,
        cols,
    }
}

impl Matrix {
    pub fn get_shape(&self) -> String {
        format!("[{} x {}]", self.rows, self.cols)
    }
    pub fn copy(&self) -> Matrix {
        let mut output_values = vec![0.0; self.rows*self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                output_values[row*self.cols+col] = self.values[row*self.cols+col];
            }
        }
        Matrix {
            values: output_values,
            rows: self.rows,
            cols: self.cols
        }
    }
    pub fn copy_to<'a>(&self, other: &'a mut Matrix) -> &'a mut Matrix {
        other.values = vec![0.0; self.rows*self.cols];
        other.rows = self.rows;
        other.cols = self.cols;
        for row in 0..self.rows {
            for col in 0..self.cols {
                other.values[row*self.cols+col] = self.values[row*self.cols+col];
            }
        }
        other
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.values[row*self.cols+col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.values[row*self.cols+col] = value;
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed_values = vec![0.0; self.rows*self.cols];
        
        for i in 0..self.rows as usize {
            for j in 0..self.cols as usize {
                transposed_values[j*self.rows+i] = self.values[i*self.cols+j];
            }
        }

        Matrix {
            values: transposed_values,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn sub_matrix(&self, start_row: usize, start_col: usize, num_rows: usize, num_cols: usize) -> Matrix {
        // Check for valid dimensions
        if (start_row + num_rows - 1)*self.cols + start_col + num_cols - 1 > self.values.len() {
            // println!("{}", self);
            panic!("Submatrix too big --\nShape: {}, len:{}, start row: {}, start col: {}, num rows: {}, num cols: {}.",
                    self.get_shape(), self.values.len(), start_row, start_col, num_rows, num_cols);
        }

        // Initialize the sub-matrix
        let mut sub_values = Vec::with_capacity(num_rows*num_cols);
        for row in 0..num_rows {
            let start = (start_row+row)*self.cols + start_col;
            let end = start + num_cols;
            sub_values.extend_from_slice(&self.values[start..end]);
        }
        Matrix {
            values: sub_values,
            rows: num_rows,
            cols: num_cols
        }
    }

    pub fn merge(a: &Matrix, b: &Matrix, c: &Matrix, d: &Matrix) -> Matrix {
        let merged_rows = a.rows + c.rows;
        let merged_cols = a.cols + b.cols;
    
        let mut merged_values = vec![0.0; merged_rows * merged_cols];
    
        for row in 0..merged_rows {
            let dest_start = row * merged_cols;
            if row < a.rows {
                let a_row_start = row * a.cols;
                merged_values[dest_start..dest_start + a.cols].copy_from_slice(&a.values[a_row_start..a_row_start + a.cols]);
    
                let b_row_start = row * b.cols;
                let b_row_slice = &b.values[b_row_start..b_row_start + b.cols];
                merged_values[dest_start + a.cols..dest_start + a.cols + b.cols].copy_from_slice(b_row_slice);
            } else {
                let c_row_start = (row - a.rows) * c.cols;
                let c_row_slice = &c.values[c_row_start..c_row_start + c.cols];
                merged_values[dest_start..dest_start + c.cols].copy_from_slice(c_row_slice);
    
                let d_row_start = (row - a.rows) * d.cols;
                let d_row_slice = &d.values[d_row_start..d_row_start + d.cols];
                merged_values[dest_start + c.cols..dest_start + c.cols + d.cols].copy_from_slice(d_row_slice);
            }
        }
    
        Matrix {
            values: merged_values,
            rows: merged_rows,
            cols: merged_cols,
        }
    }
    

    pub fn repeat(&self, num_times: usize, axis: bool) -> Matrix {

        if axis {
            // Repeat across rows
            let mut repeated_values:Vec<f32> = Vec::with_capacity(self.rows*self.cols*num_times);
            for _ in 0..num_times {
                repeated_values.extend(self.copy().values);
            }
            Matrix {
                values: repeated_values,
                rows: self.rows * num_times,
                cols: self.cols
            }
        } else {
            // Repeat across columns
            let mut repeated_values = Vec::with_capacity(self.rows*self.cols*num_times);
            for row in 0..self.rows {
                for _ in 0..num_times {
                    repeated_values.extend_from_slice(&self.values[row*self.cols..(row+1)*self.cols]);
                }
            }
            Matrix {
                values: repeated_values,
                rows: self.rows,
                cols: self.cols * num_times
            }
        }
    }

    pub fn pad_image(image: &Vec<Matrix>, padding: usize) -> Vec<Matrix> {
        let mut output = Vec::with_capacity(image.len());
        for channel in 0..image.len() {
            output.push(image[channel].pad(padding));
        }
        output
    }

    pub fn pad(&self, padding: usize) -> Matrix {
        let padded_rows = self.rows + 2*padding;
        let padded_cols = self.cols + 2*padding;
        let mut output_values = vec![0.0; (padded_rows)*(padded_cols)];
        for row in 0..self.rows {
            let input_start = row*self.cols;
            let input_end = (row+1)*self.cols;

            let output_start = row*padded_cols;
            let output_end = (row+1)*padded_cols;
            
            output_values[output_start..output_end].copy_from_slice(&self.values[input_start..input_end]);
        }

        Matrix {
            rows: padded_rows,
            cols: padded_cols,
            values: output_values
        }
    }

    pub fn one_way_pad(&self, padded_size:usize) -> Matrix {
        let mut output_values = vec![0.0; padded_size*padded_size];
        
        for row in 0..self.rows {
            let start = row * padded_size;
            let end = start + self.cols;
            let src_start = row * self.cols;
            let src_end = src_start + self.cols;
            output_values[start..end].copy_from_slice(&self.values[src_start..src_end]);
        }

        Matrix {
            values: output_values,
            rows: padded_size,
            cols: padded_size
        }
    }

    pub fn magnitude(&self) -> f32 {
        let mut magnitude = 0.0;
        for i in 0..self.values.len() {
            magnitude += self.values[i]*self.values[i];
        }
        f32::sqrt(magnitude)
    }

    pub fn add_scalar(&mut self, other: f32) -> &mut Matrix {
        for row in 0..self.rows as usize{
            for col in 0..self.cols as usize {
                self.values[row*self.cols+col] += other;
            }
        }
        self
    }

    pub fn multiply_scalar(&mut self, other: f32) -> &mut Matrix {
        for row in 0..self.rows as usize{
            for col in 0..self.cols as usize {
                self.values[row*self.cols+col] *= other;
            }
        }
        self
    }

    pub fn add(&mut self, other: &Matrix) -> &mut Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            println!("Self:\n{}\nOther:\n{}", self.get_shape(), other.get_shape());
            panic!("Matrix size mismatch for add.");
        }
        for row in 0..self.rows as usize{
            for col in 0..self.cols as usize {
                self.values[row*self.cols+col] += other.values[row*self.cols+col];
            }
        }
        self
    }

    pub fn subtract(&mut self, other: &Matrix) -> &mut Matrix {
        for row in 0..self.rows as usize {
            for col in 0..self.cols as usize {
                self.values[row*self.cols+col] -= other.values[row*self.cols+col];
            }
        }
        self
    }

    pub fn element_multiply(&mut self, other: &Matrix) -> &mut Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            println!("Self:\n{}", self);
            println!("Other:\n{}", other);
            panic!("Size mismatch for element_multiply.");
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.values[row*self.cols+col] *= other.values[row*self.cols+col];
            }
        }
        self
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        let max_dimension = std::cmp::max(std::cmp::max(self.rows, self.cols), std::cmp::max(other.rows, other.cols));
        if max_dimension < STRASSEN_CUTOFF {
            self.naive_multiply(other)
        }
        else {
            self.strassen_multiply(other, max_dimension)
        }
    }

    pub fn switch_multiply(&self, other: &Matrix, method: bool) -> Matrix {
        if !method {
            self.naive_multiply(other)
        }
        else {
            let max_dimension = std::cmp::max(std::cmp::max(self.rows, self.cols), std::cmp::max(other.rows, other.cols));
            self.strassen_multiply(other, max_dimension)
        }
    }

    fn naive_multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            println!("Self:\n{}\nInput:\n{}", self, other);
            panic!("Column and row size mismatch.");
        }
        else {
            let mut output_values = vec![0.0; self.rows*other.cols];
            for row in 0..self.rows{
                for col in 0..other.cols{
                    let mut sum:f32 = 0.0;
                    for i in 0..self.cols{
                        sum += self.values[row*self.cols+i] * other.values[i*other.cols+col];
                    }
                    output_values[row*other.cols+col] = sum;
                }
            }
            Matrix {
                values: output_values,
                rows: self.rows,
                cols: other.cols
            }
        }
    }

    //I tried, but it ended up being slower on release builds than the regular naive multiply function.
    // fn simd_multiply(&self, other: &Matrix) -> Matrix {
    //     assert_eq!(self.cols, other.rows, "Matrix dimensions do not match for multiplication.");

    //     let mut result = vec![0.0; self.rows * other.cols];

    //     for row_num in 0..self.rows {
    //         for col_num in 0..other.cols {
    //             let mut sum = 0.0;

    //             let mut index = 0;
    //             while index + SIMD_LANES < self.cols {
    //                 let row_start = row_num*self.cols;
    //                 let row_end = row_start + SIMD_LANES;
    //                 let row:Simd<f32, SIMD_LANES> = Simd::from_slice(&self.values[row_start..row_end]);

    //                 let col_start = col_num + index * other.cols;
    //                 let mut col = vec![0.0; SIMD_LANES];
    //                 for (i, val) in col.iter_mut().enumerate() {
    //                     *val = other.values[index + i * other.cols + col_num];
    //                 }
    //                 let col: Simd<f32, SIMD_LANES> = Simd::from_slice(&col);

    //                 sum += row.mul(col).reduce_sum();
    //                 index += SIMD_LANES;
    //             }
    //             while index < self.cols {
    //                 sum += self.values[row_num*self.cols+index]*other.values[index*other.cols + col_num];
    //                 index += 1;
    //             }
    //             result[row_num * other.cols + col_num] = sum;
    //         }
    //     }

    //     Matrix {
    //         values: result,
    //         rows: self.rows,
    //         cols: other.cols,
    //     }
    // }

    fn strassen_multiply(&self, other: &Matrix, max_dimension:usize) -> Matrix {
        if self.rows < 1 || self.cols < 1 || other.rows < 1 || other.cols < 1 {
            panic!("Tried to multiply 0-width matrices.");
        }
        let mut size = 1;
        while size < max_dimension {
            size <<= 1;
        }

        //Check if each subdivision is zero to avoid unnecessary calculations with all zeros.
        //Note that a and e cannot be zero with my usage, but they could be with generic matrices.
        let b_nonzero = self.cols >= size/2;
        let c_nonzero = self.rows >= size/2;
        let d_nonzero = self.cols >= size/2 && self.rows >= size/2;
        let matrix1_nonzeros = (b_nonzero, c_nonzero, d_nonzero);

        let f_nonzero = other.cols >= size/2;
        let g_nonzero = other.rows >= size/2;
        let h_nonzero = other.cols >= size/2 && other.rows >= size/2;
        let matrix2_nonzeros = (f_nonzero, g_nonzero, h_nonzero);

        //println!("Self: {}, other: {}, size: {}", self.get_shape(), other.get_shape(), size);
        let matrix1 = if std::cmp::min(self.rows, self.cols) < size {
            self.one_way_pad(size)
        } else {
            self.copy()
        };

        let matrix2 = if std::cmp::min(other.rows, other.cols) < size {
            other.one_way_pad(size)
        } else {
            other.copy()
        };
        matrix1.strassen_multiply_recursive(&matrix2, matrix1_nonzeros, matrix2_nonzeros).sub_matrix(0, 0, self.rows, other.cols)
    }

    fn strassen_multiply_recursive(&self, other: &Matrix, (b_nonzero, c_nonzero, d_nonzero): (bool, bool, bool), (f_nonzero, g_nonzero, h_nonzero): (bool, bool, bool)) -> Matrix {
        let half = self.rows/2;

        let mut a = self.sub_matrix(0, 0, half, half);

        let mut b;
        if b_nonzero {
            b = self.sub_matrix(0, half, half, half);
        }
        else {
            b = zero_matrix(half, half);
        }

        let c;
        if c_nonzero {
            c = self.sub_matrix(half, 0, half, half);
        }
        else {
            c = zero_matrix(half, half);
        }

        let d;
        if d_nonzero {
            d = self.sub_matrix(half, half, half, half);
        }
        else {
            d = zero_matrix(half, half);
        }

        let mut e = other.sub_matrix(0, 0, half, half);

        let f;
        if f_nonzero {
            f = other.sub_matrix(0, half, half, half);
        }
        else {
            f = zero_matrix(half, half);
        }

        let mut g;
        if g_nonzero {
            g = other.sub_matrix(half, 0, half, half);
        }
        else {
            g = zero_matrix(half, half);
        }

        let h;
        if h_nonzero {
            h = other.sub_matrix(half, half, half, half);
        }
        else {
            h = zero_matrix(half, half);
        }

        let mut temp1 = Matrix{
            values: vec![0.0; half*half],
            rows: 0,
            cols: 0
        };
        let mut temp2 = Matrix{
            values: vec![0.0; half*half],
            rows: 0,
            cols: 0
        };

        let method = half >= STRASSEN_CUTOFF;
        let mut p1 = a.switch_multiply(f.copy_to(&mut temp1).subtract(&h), method);
        let mut p2 = a.copy_to(&mut temp1).add(&b).switch_multiply(&h, method);
        let p3 = c.copy_to(&mut temp1).add(&d).switch_multiply(&e, method);
        let mut p4;
        if d_nonzero && g_nonzero {
            p4 = d.switch_multiply(g.copy_to(&mut temp1).subtract(&e), method);
        }
        else {
            p4 = e.copy();
            p4.multiply_scalar(-1.0);
        }
        let p5 = a.copy_to(&mut temp1).add(&d).switch_multiply(e.copy_to(&mut temp2).add(&h), method);
        let mut p6;
        if (b_nonzero || d_nonzero) && (g_nonzero || h_nonzero) {
            p6 = b.subtract(&d).switch_multiply(g.add(&h), method);
        }
        else {
            p6 = zero_matrix(half, half);
        }
        let p7 = a.subtract(&c).switch_multiply(e.add(&f), method);

        let c11 = p6.subtract(&p2).add(&p4).add(&p5);
        let c12 = p2.add(&p1);
        let c21 = p4.add(&p3);
        let c22 = p1.subtract(&p3).add(&p5).subtract(&p7);

        Matrix::merge(c11, c12, c21, c22)
    }

    pub fn im_2_col(image: &Vec<Matrix>, kernel_rows: usize, kernel_cols: usize, stride: usize, padding: usize) -> Matrix {
        let channels = image.len();
        let image_height = image[0].rows;
        let image_width = image[0].cols;

        let output_height = (image_height + 2*padding-kernel_rows) / stride + 1;
        let output_width = (image_width + 2*padding-kernel_cols) / stride + 1;

        let padded_input = Self::pad_image(image, padding);

        //Note that it's going to be transposed later.
        let column_matrix_height = channels*kernel_rows*kernel_cols;
        let column_matrix_width = output_height*output_width;
        let mut column_matrix = Matrix {
            values: vec![0.0; column_matrix_width*column_matrix_height],
            rows: column_matrix_width,
            cols: column_matrix_height
        };

        let mut row_index = 0;
        for row in 0..output_height {
            for col in 0..output_width {
                let row_start = row*stride;
                let col_start = col*stride;

                for channel in 0..channels {
                    let patch = padded_input[channel].sub_matrix(row_start, col_start, kernel_rows, kernel_cols);
                    let column_matrix_start = row_index*column_matrix_height + channel*kernel_rows*kernel_cols;
                    let column_matrix_end = column_matrix_start + kernel_rows*kernel_cols;
                    column_matrix.values[column_matrix_start..column_matrix_end].copy_from_slice(&patch.values[0..patch.cols]);
                }
                row_index+=1;
            }
        }

        column_matrix.transpose()
    }
}

pub fn zero_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix {
        values: vec![0.0; rows*cols],
        rows,
        cols
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[{} x {}]", self.rows, self.cols)?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{} ", format!("{:>6.4}", self.values[row*self.cols+col]))?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}