import random
from tic_tac_toe import minimax, get_empty_cells

class Opponent:
    def __init__(self, difficulty='easy'):
        self.difficulty = difficulty

    def make_move(self, board):
        if self.difficulty == 'easy':
            return self.easy_opponent(board)
        elif self.difficulty == 'medium':
            return self.medium_opponent(board)
        else:
            return self.hard_opponent(board)

    def easy_opponent(self, board):
        """Easy: Make a random move"""
        return random.choice(get_empty_cells(board))

    def medium_opponent(self, board):
        """Medium: Use minimax with a chance of random move"""
        if random.random() < 0.6:  # 60% chance to play optimally
            return self.hard_opponent(board)
        else:
            return self.easy_opponent(board)

    def hard_opponent(self, board):
        """Hard: Use minimax to always make the optimal move"""
        best_score = -float('inf')
        best_move = None
        for (i, j) in get_empty_cells(board):
            board[i][j] = 'O'
            score = minimax(board, 0, False)
            board[i][j] = ' '  # Undo move
            if score > best_score:
                best_score = score
                best_move = (i, j)
        return best_move
