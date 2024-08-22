use std::any::TypeId;

pub enum InputTypeEnum<InputType> {
    Single(InputType),
    Batch(Vec<InputType>),
}

pub enum OutputTypeEnum<OutputType> {
    Single(OutputType),
    Batch(Vec<OutputType>),
}

pub trait Layer<InputType, OutputType> {
    fn initialize(&mut self);
    fn pass(&self, input: InputTypeEnum<&InputType>) -> OutputTypeEnum<OutputType>;
    fn backpropagate(&mut self, input: &InputType,
            my_output: &OutputType, error: &OutputType) -> InputType;
    fn batch_backpropagate(&mut self, inputs: InputTypeEnum<&InputType>,
            my_outputs: OutputTypeEnum<&OutputType>,
            errors: OutputTypeEnum<&OutputType>) -> InputTypeEnum<InputType>;
    fn get_input_type_id(&self) -> TypeId;
    fn get_output_type_id(&self) -> TypeId;
    fn set_learning_rate(&mut self, rate: f32);
}