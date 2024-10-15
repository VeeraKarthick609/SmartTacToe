import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
        buffer_size=10000,
        batch_size=64,
        device = "cpu"
    ):
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = device

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, available_actions):
        # Exploration: choose a random action
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)  # Exploration
        else:
            # Convert the current state to a numeric representation (flattened if needed)
            state_numeric = self.state_to_numeric(state)  # Ensure this returns a flat list of 9 elements
            state_tensor = torch.FloatTensor(state_numeric).unsqueeze(0).to(self.device)  # Shape: (1, 9)

            # Get Q-values from the model
            q_values = self.model(state_tensor.unsqueeze(0))  # Assuming model outputs Q-values for each action
            q_values = q_values.squeeze().detach().cpu().numpy()  # Move back to CPU for numpy operations
            # Mask unavailable actions
            #q_values = q_values.squeeze().detach().numpy()  # Convert to numpy array
            masked_q_values = np.array(
                [q_values[i] if i in available_actions else -np.inf for i in range(9)]
            )

            # Choose the action with the highest Q-value among available actions
            return np.argmax(masked_q_values)


    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # Convert states to numeric representation
    def state_to_numeric(self,state):
        numeric_state = []
        for s in state:
            if s == "X":
                numeric_state.append(1)
            elif s == "O":
                numeric_state.append(0)
            else:
                numeric_state.append(-1)  # For empty spaces or other values
        return numeric_state

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = zip(*random.sample(self.replay_buffer, self.batch_size))

        # Convert lists to tensors and move to the appropriate device
        states = torch.FloatTensor([self.state_to_numeric(s) for s in states]).to(self.device)  # Move states to GPU
        actions = torch.LongTensor(actions).to(self.device)  # Move actions to GPU
        rewards = torch.FloatTensor(rewards).to(self.device)  # Move rewards to GPU
        next_states = torch.FloatTensor([self.state_to_numeric(s) for s in next_states]).to(self.device)  # Move next_states to GPU
        dones = torch.FloatTensor(dones).to(self.device)  # Move dones to GPU

        # Forward pass through the model
        q_values = self.model(states).gather(1, actions.view(-1, 1)).squeeze()

        # Compute target Q values
        next_q_values = self.model(next_states).max(1)[0].detach()  # Detach from graph to avoid gradients
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
