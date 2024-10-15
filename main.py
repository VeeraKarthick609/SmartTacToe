import random

# 1. Initialize an empty 3x3 board
def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

# 2. Print the board
def print_board(board):
    for row in board:
        print('|'.join(row))
        print('-' * 5)

# 3. Check if there's a winner or a draw
def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]

    return None

# 4. Get all empty cells
def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

# 5. Handle player's move
def player_move(board):
    while True:
        try:
            row, col = map(int, input("Enter row and column (0-2) separated by space: ").split())
            if (0 <= row < 3) and (0 <= col < 3) and board[row][col] == ' ':
                board[row][col] = 'X'
                break
            else:
                print("Invalid move! The cell is either occupied or out of range. Try again.")
        except ValueError:
            print("Invalid input! Please enter two numbers between 0 and 2.")

# 6. Minimax Algorithm for the AI
def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'X':  # Human wins
        return -1
    elif winner == 'O':  # AI wins
        return 1
    elif not get_empty_cells(board):  # Draw
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for (i, j) in get_empty_cells(board):
            board[i][j] = 'O'
            score = minimax(board, depth + 1, False)
            board[i][j] = ' '  # Undo move
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for (i, j) in get_empty_cells(board):
            board[i][j] = 'X'
            score = minimax(board, depth + 1, True)
            board[i][j] = ' '  # Undo move
            best_score = min(score, best_score)
        return best_score

# 7. AI's best move using Minimax
def ai_best_move(board):
    best_score = -float('inf')
    move = None
    for (i, j) in get_empty_cells(board):
        board[i][j] = 'O'
        score = minimax(board, 0, False)
        board[i][j] = ' '  # Undo move
        if score > best_score:
            best_score = score
            move = (i, j)

    if move:
        board[move[0]][move[1]] = 'O'

# 8. Main game loop
def play_game():
    board = initialize_board()
    current_player = 'X'

    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"Player {winner} wins!")
            break
        elif not get_empty_cells(board):
            print("It's a draw!")
            break

        if current_player == 'X':
            player_move(board)  # Get input from user
            current_player = 'O'
        else:
            ai_best_move(board)  # AI makes a move
            current_player = 'X'

if __name__ == "__main__":
    play_game()
