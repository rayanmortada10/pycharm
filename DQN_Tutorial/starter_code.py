# Import some modules from other libraries
import numpy as np
import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Import the environment module
from environment import Environment
from replay_buffer import ReplayBuffer

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        return 0

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions):
        # Unpack the transition tuple: (state, action, reward, next_state)
            if isinstance(transitions, tuple):
                transitions = [transitions]
            states, actions, rewards, _ = zip(*transitions)
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            predicted_q_values = q_values.gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
            target_q_values = reward_tensor
            loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)
            return loss



# Main entry point
if __name__ == "__main__":
    # Turn off the display for the environment
    environment = Environment(display=False, magnification=500)
    agent = Agent(environment)
    dqn_online = DQN()
    online_losses = []
    online_episodes = []
    for episode_num in range(100):
        agent.reset()
        episode_losses = []
        for step_num in range(20):
            transition = agent.step()
            loss = dqn_online.train_q_network(transition)
            if loss is not None:
                episode_losses.append(loss)
        if episode_losses:
            online_losses.append(sum(episode_losses) / len(episode_losses))
            online_episodes.append(episode_num)

    plt.figure()
    plt.plot(online_episodes, online_losses, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Online Learning Loss Curve')
    plt.yscale('log')
    plt.savefig("online_learning_loss.png")

    dqn_replay = DQN()
    replay_buffer = ReplayBuffer(capacity=5000)
    replay_losses = []
    replay_episodes = []
    for episode_num in range(100):
        agent.reset()
        episode_losses = []
        for step_num in range(20):
            transition = agent.step()
            replay_buffer.add(transition)
            if len(replay_buffer) >= 100:
                minibatch = replay_buffer.sample(100)
                loss = dqn_replay.train_q_network(minibatch)
                if loss is not None:
                    episode_losses.append(loss)
        if episode_losses:
            replay_losses.append(sum(episode_losses) / len(episode_losses))
            replay_episodes.append(episode_num)

    plt.figure()
    plt.plot(replay_episodes, replay_losses, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Replay Buffer Loss Curve')
    plt.yscale('log')
    plt.savefig("replay_buffer_loss.png")