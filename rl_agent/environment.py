import numpy as np
from utils.helper import Helper

class TicTacToeEnv:
    def __init__(self):
        self.board = self.initialize_board()
        self.done = False
        self.winner = None

    def initialize_board(self):
        """Initialize an empty 3x3 board"""
        return [[' ' for _ in range(3)] for _ in range(3)]

    def step(self, action, player):
        """Take an action in the environment and return the new state, reward, and done flag"""
        row, col = action
        self.board[row][col] = player
        
        winner = Helper.check_winner(self.board)
        if winner == 'X':
            self.winner = 'X'
            self.done = True
            return self.board, 1, self.done  # Positive reward for winning
        elif winner == 'O':
            self.winner = 'O'
            self.done = True
            return self.board, -1, self.done  # Negative reward for losing
        elif not Helper.get_empty_cells(self.board):
            self.done = True
            return self.board, 0, self.done  # Draw, no reward
        else:
            return self.board, 0, self.done  # Continue game, no reward yet

    def reset(self):
        """Reset the game environment"""
        self.board = self.initialize_board()
        self.done = False
        self.winner = None
        return self.board
