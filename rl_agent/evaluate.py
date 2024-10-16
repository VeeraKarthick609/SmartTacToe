from environment import TicTacToeEnv
from agent import DQNAgent
import numpy as np
from utils.helper import Helper

def evaluate_dqn(episodes=100, epsilon=0.01):  # Set epsilon for evaluation
    env = TicTacToeEnv()
    agent = DQNAgent()
    agent.load_model('weights/dqn_final_model.pth')

    wins, losses, draws = 0, 0, 0

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            available_actions = [i * 3 + j for (i, j) in Helper.get_empty_cells(state)]
            flat_state = np.array(state).flatten()

            # Include epsilon when calling choose_action
            action = agent.choose_action(flat_state, available_actions, epsilon)
            row, col = divmod(action, 3)
            state, reward, done = env.step((row, col), 'X')

            if done:
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                else:
                    draws += 1

    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

if __name__ == '__main__':
    evaluate_dqn()
