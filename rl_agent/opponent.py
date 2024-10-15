import random
from utils.helper import Helper


class Opponent:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty

    def make_move(self, board):
        if self.difficulty == "easy":
            return self.easy_opponent(board)
        elif self.difficulty == "medium":
            return self.medium_opponent(board)
        else:
            return self.hard_opponent(board)

    def easy_opponent(self, board):
        """Easy: Make a random move"""
        return random.choice(Helper.get_empty_cells(board))

    def medium_opponent(self, board):
        """Medium: Use minimax with a chance of random move"""
        if random.random() < 0.6:  # 60% chance to play optimally
            return self.hard_opponent(board)
        else:
            return self.easy_opponent(board)

    def hard_opponent(self, board):
        """Hard: Use minimax to always make the optimal move"""
        # Check if the player is about to win, block if necessary
        for i, j in Helper.get_empty_cells(board):
            board[i][j] = "X"  # Simulate player's move
            if Helper.check_winner(board) == "X":
                board[i][j] = "O"  # Block the player
                return (i, j)
            board[i][j] = " "  # Undo move

        # If no immediate block needed, find the best move
        best_move = Helper.get_best_move(board)
        if best_move:
            return best_move
        return random.choice(Helper.get_empty_cells(board))  # If no valid move found
