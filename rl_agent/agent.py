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
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 9)
        self.dropout = nn.Dropout(0.2)  # Adding dropout for regularization

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return self.fc4(x)

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
        device="cpu"
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

    def choose_action(self, state, available_actions, epsilon=0.01):
        # Epsilon-greedy strategy
        if np.random.rand() <= epsilon:
            return random.choice(available_actions)
        else:
            state_numeric = self.state_to_numeric(state)
            state_tensor = torch.FloatTensor(state_numeric).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).detach().cpu().numpy()

            # Mask unavailable actions by penalizing them heavily
            masked_q_values = np.array(
                [q_values[0][i] if i in available_actions else -1e10 for i in range(9)]
            )
            return np.argmax(masked_q_values)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def state_to_numeric(self, state):
        numeric_state = []
        for s in state:
            if s == "X":
                numeric_state.append(1)
            elif s == "O":
                numeric_state.append(-1)
            else:
                numeric_state.append(0)
        return numeric_state

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor([self.state_to_numeric(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor([self.state_to_numeric(s) for s in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get Q-values for the chosen actions from the model
        q_values = self.model(states).gather(1, actions).squeeze()

        # Double DQN: Using the main network for action selection, and the target network for Q-value evaluation
        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze().detach()

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon after replay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

# Make sure to update your training loop to pass the current epsilon value to the choose_action method.
