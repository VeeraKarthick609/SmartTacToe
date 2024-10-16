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
        """Easy: Make a random move."""
        return random.choice(Helper.get_empty_cells(board))

    def medium_opponent(self, board):
        """Medium: Use minimax with a chance of random move."""
        if random.random() < 0.6:  # 60% chance to play optimally
            return self.hard_opponent(board)
        else:
            return self.easy_opponent(board)

    def hard_opponent(self, board):
        """Hard: Use minimax to play optimally with a chance of randomness."""
        if random.random() < 0.2:  # 20% chance to make a random (suboptimal) move
            return random.choice(Helper.get_empty_cells(board))

        best_score = float('-inf')
        best_move = None

        for move in Helper.get_empty_cells(board):
            row, col = move
            board[row][col] = "O"  # Simulate opponent's move

            # Evaluate the move using minimax
            score = self.minimax(board, depth=0, is_maximizing=False, alpha=float('-inf'), beta=float('inf'))

            board[row][col] = " "  # Undo the move

            if score > best_score:
                best_score = score
                best_move = (row, col)

        return best_move

    def minimax(self, board, depth, is_maximizing, alpha, beta):
        # Check for terminal conditions
        if self.check_win(board, "O"):
            return 1  # Opponent wins
        elif self.check_win(board, "X"):
            return -1  # Player wins
        elif not Helper.get_empty_cells(board):
            return 0  # Draw

        if is_maximizing:
            max_eval = float('-inf')
            for move in Helper.get_empty_cells(board):
                row, col = move
                board[row][col] = "O"  # Simulate opponent's move
                eval = self.minimax(board, depth + 1, False, alpha, beta)
                board[row][col] = " "  # Undo the move
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in Helper.get_empty_cells(board):
                row, col = move
                board[row][col] = "X"  # Simulate player's move
                eval = self.minimax(board, depth + 1, True, alpha, beta)
                board[row][col] = " "  # Undo the move
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def check_win(self, board, player):
        """Check rows, columns, and diagonals for a win."""
        # Check rows
        for row in range(3):
            if all([cell == player for cell in board[row]]):
                return True
        # Check columns
        for col in range(3):
            if all([board[row][col] == player for row in range(3)]):
                return True
        # Check diagonals
        if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
            return True
        return False
