use std::f32::consts::TAU;

pub fn normal_pdf(z: f32, mean: f32, std_dev: f32) -> f32 {
    let output = (-(z - mean)*(z - mean)/(2.*std_dev*std_dev)).exp() / (std_dev * TAU.sqrt());
    //println!("\t\tPDF: {}", output);
    output
}

pub fn standard_normal_pdf(z: f32) -> f32 {
    (-z*z/2.).exp() / TAU.sqrt()
}

pub fn normal_cdf(z: f32, mean: f32, std_dev: f32) -> f32 {
    standard_normal_cdf((z - mean) / std_dev)
}

pub fn standard_normal_cdf(z: f32) -> f32 {
    let start = z - 50.;
    let step_size = 0.1;
    let mut sum = 0.;
    // println!("\tStart: {}, z: {}, step size: {}", start, z, step_size);
    let mut x = start;
    while x <= z {
        sum += normal_pdf(x, 0., 1.) * step_size;
        x += step_size;
    }

    sum
}