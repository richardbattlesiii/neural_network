pub trait Layer<'a> {
    type Input: 'a;
    type Output;
    type MyOutputAsAnInput: 'a;
    type MyInputAsAnOutput;

    fn initialize(&mut self);
    fn set_learning_rate(&mut self, rate: f32);
    fn pass(&self, input: &'a Self::Input) -> Self::Output;
    fn backpropagate(&mut self, layer_input: &'a Self::Input,
            layer_output: &'a Self::MyOutputAsAnInput,
            dl_da: &'a Self::MyOutputAsAnInput) -> Self::MyInputAsAnOutput;
}