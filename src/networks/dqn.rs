use crate::{environments::environment::Environment, SoftmaxLayer};
use crate::networks::neural_net::NeuralNet;
use crate::layers::dense_layer::DenseLayer;
use crate::environments::cart_pole::CartPole;
use ndarray::{Array1, Array2, Array3, ArrayD, ArrayViewD};
use rand::Rng;
use rand::thread_rng;
use std::f32::consts::PI;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::{LineJoin, LineStyle, PointMarker, PointStyle};

pub struct DQN {
    policy_network: NeuralNet,
    target_network: NeuralNet,
    epsilon: f32,
    epsilon_decay: f32,
    min_epsilon: f32,
    gamma: f32,
    num_actions: usize,
    memory: Vec<(ArrayD<f32>, usize, f32, ArrayD<f32>)>,
    sample_size: usize,
    buffer_size: usize,
}

pub fn train_dqn_on_cart_pole() {
    let episodes = 10000;
    let mut rewards_per_episode: Vec<(f64, f64)> = Vec::with_capacity(episodes);
    let max_steps_per_episode = 600;
    let num_inputs = 4; //State size
    let num_actions = 2; //Number of actions: left, right
    let mut dqn = DQN::new(num_inputs, num_actions);
    let mut rng = thread_rng();
    
    for episode in 0..episodes {
        let mut env = CartPole::new();
        let mut state = env.get_state().into_dyn();

        let mut total_reward = 0.0;
        for _ in 0..max_steps_per_episode {
            let action = dqn.choose_action(&state, &mut rng);
            //println!("{}", action);
            env.update(action as u16); //Update environment based on chosen action
            
            let reward = env.reward();
            total_reward += reward;

            let next_state = env.get_state().into_dyn();
            
            let done = env.is_done();

            dqn.store_experience(state, action, reward, next_state.clone());

            state = next_state;

            if done {
                break;
            }
        }
            
        dqn.train();

        if episode % 2 == 0 {
            dqn.target_network = dqn.policy_network.clone();
        }

        if episode % 100 == 0 {
            println!("Episode {}: Total Reward: {:6.2} Epsilon: {:5.3}", episode, total_reward, dqn.epsilon);
        }
        rewards_per_episode.push((episode as f64, total_reward as f64));
    }

    let s1: Plot = Plot::new(rewards_per_episode)
        .line_style(
            LineStyle::new()
                .linejoin(LineJoin::Miter)
                .width(0.5)
                .colour("#3080FF")
            );

    let v = ContinuousView::new().add(s1);
    Page::single(&v).save("scatter.svg").unwrap();
}

impl DQN {
    pub fn new(num_inputs: usize, num_actions: usize) -> Self {
        let mut nn = NeuralNet::new();
        nn.add_layer(Box::new(DenseLayer::new(
            num_inputs, //input size
            16, // output size
            0.001, // learning_rate
            0.1,   // lambda (regularization term)
            1,     // ReLU activation
        )));

        nn.add_layer(Box::new(DenseLayer::new(
            16, // input size
            num_actions, // output size
            0.001, // learning rate
            0.1, //lambda (regularization term)
            0, // Linear activation
        )));

        nn.initialize();
        
        DQN {
            policy_network: nn.clone(),
            target_network: nn,
            epsilon: 1.0,
            epsilon_decay: 0.99975,
            min_epsilon: 0.01,
            gamma: 0.9,
            num_actions,
            memory: vec![],
            sample_size: 256,
            buffer_size: 256,
        }
    }

    pub fn choose_action(&self, state: &ArrayD<f32>, rng: &mut impl Rng) -> usize {
        if rng.gen::<f32>() < self.epsilon {
            rng.gen_range(0..self.num_actions)
        } else {
            let q_values = self.policy_network.predict(state);
            //println!("{}", q_values);
            let mut max = f32::MIN;
            let mut max_index = usize::MAX;
            for i in 0..q_values.len() {
                if q_values[[0, i]] > max {
                    max = q_values[[0, i]];
                    max_index = i;
                }
            }
            max_index
        }
    }

    pub fn store_experience(&mut self, state: ArrayD<f32>, action: usize, reward: f32, next_state: ArrayD<f32>) {
        self.memory.push((state, action, reward, next_state));
        if self.memory.len() > self.buffer_size {
            self.memory.remove(0);
        }
    }

    fn sample_experiences(&self, num_experiences: usize) -> Vec<(ArrayD<f32>, usize, f32, ArrayD<f32>)>{
        if self.memory.len() < num_experiences {
            return vec![];
        }
        let mut memories_remaining: Vec<usize> = (0..self.memory.len()).collect();
        let mut memories_to_use: Vec<usize> = Vec::with_capacity(num_experiences);
        let mut rng = rand::thread_rng();
        for _ in 0..num_experiences {
            let index_of_memory: usize = rng.gen_range(0..memories_remaining.len());
            memories_to_use.push(memories_remaining[index_of_memory]);
            memories_remaining.swap_remove(index_of_memory);
        }

        let mut output: Vec<(ArrayD<f32>, usize, f32, ArrayD<f32>)> = Vec::with_capacity(num_experiences);
        for i in 0..num_experiences {
            output.push(self.memory[memories_to_use[i]].clone());
        }
        output
    }

    pub fn train(&mut self) {
        let experiences = self.sample_experiences(self.sample_size);
        if experiences.len() < self.sample_size {
            return;
        }
        
        for experience in experiences {
            let (state, action, reward, next_state) = experience;

            let mut max_next_q_value = f32::MIN;
            let q_values = self.target_network.predict(&next_state);
            for i in 0..q_values.len() {
                let value = q_values[[0, i]];
                if value > max_next_q_value {
                    max_next_q_value = value;
                }
            }
            let target_value = reward + self.gamma * max_next_q_value;
            
            let current_q_values = self.policy_network.predict(&state.clone());
            let mut target_q_values = current_q_values.clone();
            target_q_values[[0, action]] = target_value;

            self.policy_network.accumulate_gradients(&state, &target_q_values, self.num_actions);
        }

        self.policy_network.apply_gradients();

        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }
}
