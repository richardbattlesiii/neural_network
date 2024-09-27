use ndarray::{ArrayD, ArrayViewD};

pub trait Layer {
    ///Uses exclusively Xavier initialization for now.
    fn initialize(&mut self);
    fn set_learning_rate(&mut self, rate: f32);
    fn pass(&self, input: &ArrayD<f32>) -> ArrayD<f32>;
    ///Calculates and applies gradients for a single batch.
    ///Note that this zeroes the gradients before applying the new ones.
    fn backpropagate(
            &mut self,
            layer_input: &ArrayD<f32>,
            layer_output: &ArrayD<f32>,
            dl_da: &ArrayD<f32>)
            -> ArrayD<f32>;
    fn get_input_shape(&self) -> Vec<usize>;
    fn get_output_shape(&self) -> Vec<usize>;
    ///Returns a deep copy of the layer in a box.
    fn copy_into_box(&self) -> Box<dyn Layer>;
    ///Sets the gradients and `self.num_batches` to 0.
    fn zero_gradients(&mut self);
    ///Calculates gradients without applying them yet and increments `self.num_batches`.
    fn accumulate_gradients(
        &mut self,
        layer_input: &ArrayD<f32>,
        layer_output: &ArrayD<f32>,
        dl_da: &ArrayD<f32>)
        -> ArrayD<f32>;
    ///Applies the average of gradients from `accumulate_gradients`
    ///and returns how many batches there were.
    fn apply_accumulated_gradients(&mut self);
}