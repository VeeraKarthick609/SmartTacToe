import os
import matplotlib.pyplot as plt
from environment import TicTacToeEnv
from agent import DQNAgent
from opponent import Opponent
import numpy as np
from utils.helper import Helper
import torch


# Convert states to numeric representation
def state_to_numeric(state):
    numeric_state = []
    for s in state:
        if s == "X":
            numeric_state.append(1)
        elif s == "O":
            numeric_state.append(0)
        else:
            numeric_state.append(-1)  # For empty spaces or other values
    return numeric_state


def train_dqn(agent, episodes=10000, difficulty="medium", device="cpu"):
    env = TicTacToeEnv()
    opponent = Opponent(difficulty=difficulty)

    rewards = []  # List to store rewards per episode
    win_counts = []  # Count of wins per episode

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0  # Initialize total reward for the episode

        while not done:
            available_actions = [i * 3 + j for (i, j) in Helper.get_empty_cells(state)]
            flat_state = np.array(state).flatten()
            flat_state_tensor = torch.FloatTensor(state_to_numeric(flat_state)).to(device)

            # Agent plays
            action = agent.choose_action(flat_state_tensor, available_actions)
            row, col = divmod(action, 3)
            next_state, reward, done = env.step((row, col), "X")

            total_reward += reward  # Update total reward for the episode

            if done:
                agent.store_experience(
                    flat_state, action, reward, np.array(next_state).flatten(), done
                )
                win_counts.append(1 if reward == 1 else 0)  # Append win count
                break

            # Opponent plays
            opponent_move = opponent.make_move(env.board)
            next_state, _, done = env.step(opponent_move, "O")

            if done:
                reward = -1  # Loss for the agent
                agent.store_experience(
                    flat_state, action, reward, np.array(next_state).flatten(), done
                )
                win_counts.append(0)  # Append loss count
                break

            # Store the experience and train the agent
            agent.store_experience(
                flat_state, action, reward, np.array(next_state).flatten(), done
            )
            agent.replay()

        rewards.append(total_reward)  # Append total reward for the episode

        # Print the episode details
        print(f"Difficulty: {difficulty}, Episode {episode + 1}/{episodes}: Total Reward = {total_reward}, Wins = {sum(win_counts)}")

        # Update the target model periodically
        if episode % 100 == 0:
            agent.update_target_model()
            print(f"Updated target model at episode {episode}.")

    # After training, plot the rewards and other visualizations
    plot_rewards(rewards, difficulty)
    plot_cumulative_rewards(rewards, difficulty)
    plot_win_rate(win_counts, difficulty)
    plot_moving_average(rewards, difficulty)

def plot_rewards(rewards, difficulty, filename=None):
    os.makedirs("graphs", exist_ok=True)
    if filename is None:
        filename = f"graphs/rewards_plot_{difficulty}.png"
    plt.figure()
    plt.plot(rewards)
    plt.title(f"Rewards per Episode - Difficulty: {difficulty}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig(filename)
    plt.close()

def plot_cumulative_rewards(rewards, difficulty):
    cumulative_rewards = np.cumsum(rewards)
    plt.figure()
    plt.plot(cumulative_rewards)
    plt.title(f"Cumulative Rewards - Difficulty: {difficulty}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.savefig(f"graphs/cumulative_rewards_plot_{difficulty}.png")
    plt.close()

def plot_win_rate(win_counts, difficulty):
    win_rate = np.array(win_counts) / np.arange(1, len(win_counts) + 1) * 100
    plt.figure()
    plt.plot(win_rate)
    plt.title(f"Win Rate over Episodes - Difficulty: {difficulty}")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.grid()
    plt.savefig(f"graphs/win_rate_plot_{difficulty}.png")
    plt.close()

def plot_moving_average(rewards, difficulty, window_size=100):
    moving_average = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.figure()
    plt.plot(moving_average)
    plt.title(f"Moving Average of Rewards (window size = {window_size}) - Difficulty: {difficulty}")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.grid()
    plt.savefig(f"graphs/moving_average_plot_{difficulty}.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a single DQNAgent
    agent = DQNAgent(device=device)

    # Define difficulty levels and corresponding episode counts
    difficulties = ["easy", "medium", "hard"]
    episodes = [5000, 10000, 20000]

    # Create 'weights' directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)

    # Train for each difficulty level
    for difficulty, episode_count in zip(difficulties, episodes):
        print(f"Training with difficulty: {difficulty}, episodes: {episode_count}...")
        train_dqn(agent, episodes=episode_count, difficulty=difficulty, device=device)

    # Save the final model after training with all difficulties
    agent.save_model("weights/dqn_final_model.pth")
    print("Final model saved in 'weights' folder.")
