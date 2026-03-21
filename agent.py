############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Deep Q-Network for state-action value function approximation
    Maps continuous 2D state (position) to Q-value
    """

    def __init__(self, state_dim=2, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single Q-value output
        )

    def forward(self, state):
        """
        Forward pass: state -> Q-value
        Args:
            state: tensor of shape [batch_size, 2] or [2]
        Returns:
            q_value: tensor of shape [batch_size, 1] or [1]
        """
        return self.net(state)


class ExperienceReplayBuffer:
    """
    Experience Replay Memory Bank
    Stores transitions and allows random sampling to decorrelate training data
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        """
        Store experience: (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Randomly sample batch to break temporal correlations
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class Agent:
    """
    Deep DQN Agent with:
    - Model-free control (learning Q-values)
    - Experience replay (memory bank)
    - Double Q-Learning (two networks)
    - Epsilon-greedy exploration with GLIE decay
    - Optimized reward engineering (-1 step penalty)
    """

    def __init__(self):
        """
        Initialize agent with neural networks, replay buffer, and exploration strategy
        """
        # ==================== EPISODE PARAMETERS ====================
        self.episode_length = 500  # Steps per episode
        self.num_steps_taken = 0  # Total training steps
        self.episodes_completed = 0  # Episode counter

        # Current transition tracking
        self.state = None
        self.action = None

        # ==================== DEVICE CONFIGURATION ====================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Using device: {self.device}")

        # ==================== NEURAL NETWORK ARCHITECTURE ====================
        # Q-network: learns to predict Q(s)
        self.q_network = QNetwork(state_dim=2, hidden_dim=256).to(self.device)

        # Target network: separate network for stability (Double Q-Learning)
        self.target_network = QNetwork(state_dim=2, hidden_dim=256).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode

        # ==================== OPTIMIZER ====================
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # ==================== EXPERIENCE REPLAY BUFFER ====================
        self.replay_buffer = ExperienceReplayBuffer(capacity=100000)
        self.batch_size = 32
        self.min_buffer_size = 1000  # Wait for buffer to fill before training

        # ==================== REWARD ENGINEERING ====================
        # Step penalty: -1 for each step creates "sense of urgency"
        self.step_penalty = -1.0

        # Goal reward: large positive reward for reaching goal
        self.goal_reward = 10.0

        # Distance threshold for goal
        self.goal_threshold = 0.03

        # ==================== DEEP Q-LEARNING PARAMETERS ====================
        self.gamma = 0.99  # Discount factor - values future rewards
        self.tau = 0.001  # Soft update coefficient for target network
        self.update_target_frequency = 500  # Hard update every N steps

        # ==================== EXPLORATION STRATEGY (GLIE) ====================
        # Epsilon-greedy: P(random action) = epsilon, P(greedy) = 1 - epsilon
        self.epsilon = 0.5  # Start with full exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9  # Decay per episode (GLIE schedule)

        # Action clipping
        self.max_action = 0.02

        # ==================== TRAINING METRICS ====================
        self.training_step = 0
        self.episode_rewards = []
        self.episode_steps = []

    # ========================================================================
    # REQUIRED INTERFACE FUNCTIONS
    # ========================================================================

    def has_finished_episode(self):
        """
        Check if current episode has finished based on episode length
        """
        if self.num_steps_taken > 0 and self.num_steps_taken % self.episode_length == 0:
            return True
        return False

    def get_next_action(self, state):
        """
        TRAINING ACTION: Epsilon-greedy exploration strategy
        - With probability epsilon: take random action (exploration)
        - With probability 1-epsilon: take best known action (exploitation)

        Args:
            state: numpy array [2] representing agent position
        Returns:
            action: numpy array [2] representing continuous 2D movement
        """
        # Update step counter
        self.num_steps_taken += 1
        self.state = state

        # ==================== EPSILON-GREEDY DECISION ====================
        if np.random.random() < self.epsilon:
            # EXPLORATION: Random action
            action = np.random.uniform(
                low=-self.max_action,
                high=self.max_action,
                size=2
            ).astype(np.float32)
        else:
            # EXPLOITATION: Greedy action from Q-network
            action = self._compute_greedy_action(state)

        self.action = action

        # ==================== TRAINING UPDATE ====================
        # Only train if we have enough experience in buffer
        if len(self.replay_buffer) >= self.min_buffer_size:
            # Train network on random batch
            self._train_on_batch()

            # Periodically hard-update target network
            if self.training_step % self.update_target_frequency == 0:
                self._hard_update_target_network()

        self.training_step += 1
        return action

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        """
        REWARD ENGINEERING: Process transition and compute reward

        Reward structure (humanized):
        - Base: -1 per step (creates urgency)
        - Goal bonus: +10 if distance < threshold
        - Distance incentive: +1 - distance (move closer)

        Args:
            next_state: numpy array [2] of agent's new position
            distance_to_goal: float, Euclidean distance to goal
        """
        # ==================== REWARD COMPUTATION ====================
        # Start with step penalty (-1 for each step)
        reward = self.step_penalty

        # Add distance-based incentive (closer = higher reward)
        reward += (1.0 - distance_to_goal)

        # Large bonus if goal reached
        if distance_to_goal < self.goal_threshold:
            reward += self.goal_reward

        # ==================== EXPERIENCE STORAGE ====================
        # Create transition: (state, action, reward, next_state, done)
        done = distance_to_goal < self.goal_threshold
        transition = (self.state, self.action, reward, next_state, done)

        # Store in replay buffer (Memory Bank)
        self.replay_buffer.add(transition)

        # ==================== EPISODE TRACKING ====================
        if self.has_finished_episode():
            self.episodes_completed += 1

            # GLIE: Decay epsilon for transition from exploration to exploitation
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f"[Episode {self.episodes_completed}] "
                  f"Epsilon: {self.epsilon:.4f}, "
                  f"Buffer: {len(self.replay_buffer)}")

    def get_greedy_action(self, state):
        """
        TESTING ACTION: Pure exploitation (greedy policy)
        Used during 100-step test phase - no exploration

        Args:
            state: numpy array [2] representing agent position
        Returns:
            action: numpy array [2] with highest Q-value
        """
        return self._compute_greedy_action(state)

    # ========================================================================
    # INTERNAL HELPER FUNCTIONS
    # ========================================================================

    def _compute_greedy_action(self, state):
        """
        Compute greedy action by feeding state through Q-network
        and using gradient-based optimization to find action that maximizes Q-value
        """
        # Initialize action as learnable parameter
        action = torch.zeros(2, dtype=torch.float32,
                             requires_grad=True, device=self.device)

        # Optimizer for action
        action_optimizer = torch.optim.Adam([action], lr=0.05)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Optimize action to maximize Q-value
        for _ in range(5):  # 5 inner optimization steps
            # Predict next state
            next_state = state_tensor + action.unsqueeze(0)
            next_state = torch.clamp(next_state, 0.0, 1.0)

            # Get Q-value for this action
            q_value = self.q_network(next_state)

            # Loss: negative Q-value (we want to maximize)
            loss = -q_value

            # Gradient update
            action_optimizer.zero_grad()
            loss.backward()
            action_optimizer.step()

        # Extract final action and clip to valid range
        action_final = torch.clamp(
            action.detach(),
            -self.max_action,
            self.max_action
        )

        return action_final.cpu().numpy().astype(np.float32)

    def _train_on_batch(self):
        """
        Deep Q-Learning Training Step:
        - Sample random batch from replay buffer (break correlations)
        - Compute Bellman target using target network (Double Q-Learning)
        - Update Q-network via gradient descent
        """
        # ==================== BATCH SAMPLING ====================
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to PyTorch tensors
        state_tensor = torch.FloatTensor(states).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ==================== CURRENT Q-VALUES ====================
        # Q(s, a) from main network
        current_q_values = self.q_network(state_tensor)

        # ==================== TARGET Q-VALUES (Double Q-Learning) ====================
        # Use target network for stability (decouple action selection from evaluation)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)

            # Bellman backup: Q(s, a) = r + gamma * max Q(s', a')
            target_q_values = reward_tensor + self.gamma * next_q_values * (1 - done_tensor)

        # ==================== LOSS COMPUTATION ====================
        # Mean Squared Error between predicted and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # ==================== BACKPROPAGATION ====================
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

    def _hard_update_target_network(self):
        """
        Hard update: Copy weights from Q-network to target network
        Done periodically to provide stable training targets
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
