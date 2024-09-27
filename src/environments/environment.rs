use ndarray::Array3;

pub trait Environment {
    fn update(&mut self, action: u16);
    fn get_state(&self) -> Array3<f32>;
    fn reward(&self) -> f32;
    fn num_actions(&self) -> usize;
    fn is_done(&self) -> bool;
}