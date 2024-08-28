use ndarray::prelude::*;
use num::complex::*;
use std::f32::consts::TAU;
use rayon::prelude::*;

fn fft_1d(input: &ArrayView1<Complex32>) -> Vec<Complex32> {
    //println!("\tFFT1d input:\n\t{}", input);
    let len = input.len();

    if len == 1 {
        return vec![input[[0]]];
    }

    let even = fft_1d(&input.slice(s![0..len;2]));
    let odd = fft_1d(&input.slice(s![1..len;2]));

    let mut output: Vec<Complex32> = vec![Complex32::default(); len];
    
    (0..len/2).into_iter().for_each(|k| {
        let wk = c32(0., -TAU * k as f32 / len as f32).exp();
        let t = wk * odd[k];
        output[k] = even[k] + t;
        output[k + len/2] = even[k] - t;
    });

    //println!("\tFFT1d output:\n\t{:?}", output);
    output
}

pub fn fft(input: &ArrayView2<f32>) -> Array2<Complex32> {
    let rows = input.nrows();
    let cols = input.ncols();
    //println!("FFT Input:\n{}", input);
    let complex_input = input.map(|real_num| {
        c32(*real_num, 0.)
    });
    let rows_fftd: Vec<Complex32> = (0..rows)
            .into_iter()
            .flat_map(|row| {
                fft_1d(&complex_input.slice(s![row, ..]))
    })
    .collect();

    let temp = Array2::from_shape_vec((rows, cols), rows_fftd).unwrap();
    
    let cols_fftd: Vec<Complex32> = (0..cols)
            .into_iter()
            .flat_map(|col| {
                fft_1d(&temp.slice(s![.., col]))
    })
    .collect();

    Array2::from_shape_vec((rows, cols), cols_fftd).unwrap()
}



fn ifft_1d(input: &ArrayView1<Complex32>) -> Vec<Complex32> {
    let len = input.len();

    if len == 1 {
        return vec![input[[0]]];
    }

    let even = fft_1d(&input.slice(s![0..len;2]));
    let odd = fft_1d(&input.slice(s![1..len;2]));

    let mut output: Vec<Complex32> = vec![Complex32::default(); len];
    
    (0..len/2).into_iter().for_each(|k| {
        let wk = c32(0., TAU * k as f32 / len as f32).exp();
        let t = wk * odd[k];
        output[k] = even[k] + t;
        output[k + len/2] = even[k] - t;
    });

    //Normalize
    output.iter_mut().for_each(|x| *x /= len as f32); 

    output
}

pub fn ifft(input: &Array2<Complex32>) -> Array2<f32> {
    let rows = input.nrows();
    let cols = input.ncols();
    
    let cols_ifftd: Vec<Complex32> = (0..rows)
            .into_iter()
            .flat_map(|row| {
                ifft_1d(&input.slice(s![row, ..]))
    })
    .collect();

    let temp = Array2::from_shape_vec((rows, cols), cols_ifftd).unwrap();

    //println!("Temp:\n{}", temp);
    //println!("Temp slice:\n{}", temp.slice(s![.., 0]));
    let rows_ifftd: Vec<Complex32> = (0..cols)
            .into_iter()
            .flat_map(|col| {
                ifft_1d(&temp.slice(s![.., col]))
    })
    .collect();

    //println!("Rows fftd:\n{:?}", rows_ifftd);

    let real_parts: Vec<f32> = shift(&rows_ifftd, rows, cols)
            .into_iter()
            .map(|complex_num| {
                return complex_num.re;
    }).collect();

    Array2::from_shape_vec((rows, cols), real_parts).unwrap()
}

fn shift(arr: &Vec<Complex32>, rows: usize, cols: usize) -> Array2<Complex32> {
    let mut shifted = Array2::zeros((rows, cols));
    let row_shift = rows - 1;
    let col_shift = cols - 1;
    
    for r in 0..rows {
        for c in 0..cols {
            let new_r = (r + row_shift) % rows;
            let new_c = (c + col_shift) % cols;
            shifted[[new_r, new_c]] = arr[r*cols + c];
        }
    }
    
    shifted
}

pub fn pad(input: &ArrayView2<f32>, padded_size: usize) -> Array2<f32> {
    let input_size = input.nrows();
    if input_size > padded_size {
        panic!("Image was bigger than the padded size.");
    }
    else if input_size == padded_size {
        return input.to_owned();
    }
    else {
        let difference = (padded_size - input_size) / 2;
        let mut output:Array2<f32> = Array2::zeros((padded_size, padded_size));
        output.slice_mut(s![difference..input_size+difference, difference..input_size+difference]).assign(input);
        output
    }
}

///This is way, way slower than the basic approach. I love optimization!!!
pub fn convolve(image: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
    //It's that easy!
    let padded_kernel = pad(kernel, image.nrows());
    //println!("Padded kernel:\n{}", padded_kernel);
    let fft_image = fft(image);
    let fft_kernel = fft(&padded_kernel.view());
    //println!("FFT Image:\n{}\nFFT Kernel:\n{}", fft_image, fft_kernel);
    let fft_result = &fft_image * &fft_kernel;
    //println!("FFT Result:\n{}", fft_result);
    let result = ifft(&fft_result);
    result
}