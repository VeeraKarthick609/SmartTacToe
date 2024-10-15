import random
from dataclasses import dataclass

@dataclass
class Helper:
    @staticmethod
    def initialize_board():
        return [[' ' for _ in range(3)] for _ in range(3)]

    @staticmethod
    def check_winner(board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != ' ':
                return board[i][0]
            if board[0][i] == board[1][i] == board[2][i] != ' ':
                return board[0][i]
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return board[0][2]
        if all(board[i][j] != ' ' for i in range(3) for j in range(3)):
            return 'Draw'
        return None

    @staticmethod
    def get_empty_cells(board):
        return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

    @classmethod
    def minimax(cls, board, depth, is_maximizing):
        winner = cls.check_winner(board)
        if winner == 'X':
            return -10 + depth
        elif winner == 'O':
            return 10 - depth
        elif winner == 'Draw':
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for (i, j) in cls.get_empty_cells(board):
                board[i][j] = 'O'
                score = cls.minimax(board, depth + 1, False)
                board[i][j] = ' '
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for (i, j) in cls.get_empty_cells(board):
                board[i][j] = 'X'
                score = cls.minimax(board, depth + 1, True)
                board[i][j] = ' '
                best_score = min(score, best_score)
            return best_score

    @classmethod
    def ai_best_move(cls, board, difficulty):
        if difficulty == "Easy":
            move = random.choice(cls.get_empty_cells(board))
        elif difficulty == "Medium":
            if random.random() < 0.6:
                move = cls.get_best_move(board)
            else:
                move = random.choice(cls.get_empty_cells(board))
        else:  # Hard
            move = cls.get_best_move(board)
        
        board[move[0]][move[1]] = 'O'
        return board

    @classmethod
    def get_best_move(cls, board):
        best_score = -float('inf')
        best_move = None
        for (i, j) in cls.get_empty_cells(board):
            board[i][j] = 'O'
            score = cls.minimax(board, 0, False)
            board[i][j] = ' '
            if score > best_score:
                best_score = score
                best_move = (i, j)
        return best_move