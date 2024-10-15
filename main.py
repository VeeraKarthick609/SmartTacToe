import streamlit as st
import random

# Initialize an empty 3x3 board
def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

# Check if there's a winner or a draw
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

# Get all empty cells
def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

# Minimax Algorithm for the AI
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

# AI's best move using Minimax
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

# Main game function
def main():
    st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮", layout="centered")
    
    st.title("🎮 Tic-Tac-Toe Challenge 🤖")

    # Initialize session state to store game data across interactions
    if "board" not in st.session_state:
        st.session_state.board = initialize_board()
        st.session_state.current_player = "X"
        st.session_state.winner = None

    # Display the current board
    st.write("### Current Board:")
    
    cols = st.columns(3)  # Create 3 columns for the board
    for i in range(3):
        for j in range(3):
            with cols[j]:
                if st.session_state.board[i][j] == "X":
                    st.button("❌", key=f"{i}-{j}", disabled=True)
                elif st.session_state.board[i][j] == "O":
                    st.button("⭕", key=f"{i}-{j}", disabled=True)
                else:
                    if st.button("🔲", key=f"{i}-{j}") and st.session_state.winner is None:
                        st.session_state.board[i][j] = "X"
                        st.session_state.current_player = "O"  # Switch to AI
                        if check_winner(st.session_state.board):
                            st.session_state.winner = "X"  # Player wins
                            st.session_state.player_wins += 1
                        elif not get_empty_cells(st.session_state.board):
                            st.session_state.winner = "Draw"
                            st.session_state.draws += 1
                        else:
                            ai_best_move(st.session_state.board)  # AI plays
                            if check_winner(st.session_state.board):
                                st.session_state.winner = "O"  # AI wins
                                st.session_state.ai_wins += 1
                            elif not get_empty_cells(st.session_state.board):
                                st.session_state.winner = "Draw"
                                st.session_state.draws += 1
                        st.rerun()

    # Display winner or draw
    if st.session_state.winner:
        if st.session_state.winner == "X":
            st.success("🎉 You win! Congratulations! 🏆")
        elif st.session_state.winner == "O":
            st.error("🤖 AI wins! Better luck next time! 💻")
        else:
            st.info("🤝 It's a draw! Great minds think alike! 🧠")

    # Restart the game
    if st.button("🔄 New Game", key="restart"):
        st.session_state.board = initialize_board()
        st.session_state.current_player = "X"
        st.session_state.winner = None

if __name__ == "__main__":
    main()