import numpy as np
from utils.helper import Helper

class TicTacToeEnv:
    def __init__(self):
        self.board = self.initialize_board()
        self.done = False
        self.winner = None

    def initialize_board(self):
        """Initialize an empty 3x3 board."""
        return [[' ' for _ in range(3)] for _ in range(3)]

    def step(self, action, player):
        """Take an action in the environment and return the new state, reward, and done flag."""
        row, col = action

        # Place the player's mark on the board
        self.board[row][col] = player

        # Check for a winner
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
            return self.board, 0, self.done  # Draw, neutral reward
        else:
            # Implement logic to give a penalty if the agent is close to losing
            if self.is_near_loss(player):
                return self.board, -1, self.done  # Penalty for being in a losing position

            return self.board, 0, self.done  # Continue game, no reward yet

    def is_near_loss(self, player):
        """Check if the player is one move away from losing."""
        opponent = 'O' if player == 'X' else 'X'
        for move in Helper.get_empty_cells(self.board):
            row, col = move
            self.board[row][col] = opponent
            if Helper.check_winner(self.board) == opponent:
                self.board[row][col] = ' '  # Undo the move
                return True
            self.board[row][col] = ' '  # Undo the move
        return False

    def reset(self):
        """Reset the game environment."""
        self.board = self.initialize_board()
        self.done = False
        self.winner = None
        return self.board

    def render(self):
        """Display the current state of the board."""
        for row in self.board:
            print("|".join(row))
            print("-" * 5)

