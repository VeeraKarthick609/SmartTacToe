import streamlit as st
from utils.helper import Helper  # Assuming the Helper class is in a file named tic_tac_toe_helper.py

def main():
    st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮", layout="centered")
    st.title("🎮 Tic-Tac-Toe Challenge 🤖")

    # Difficulty selection
    difficulty = st.selectbox("Select AI Difficulty Level:", ["Easy", "Medium", "Hard"])

    # Initialize session state to store game data across interactions
    if "board" not in st.session_state:
        st.session_state.board = Helper.initialize_board()
        st.session_state.current_player = "X"
        st.session_state.winner = None
        st.session_state.player_wins = 0
        st.session_state.ai_wins = 0
        st.session_state.draws = 0

    # Display player stats
    st.write(f"🏆 Player Wins: {st.session_state.player_wins} | AI Wins: {st.session_state.ai_wins} | Draws: {st.session_state.draws}")

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
                        make_move(i, j, difficulty)
                        st.rerun()

    # Display winner or draw
    if st.session_state.winner:
        display_result()

    # Restart the game
    if st.button("🔄 New Game", key="restart"):
        reset_game()

def make_move(i, j, difficulty):
    st.session_state.board[i][j] = "X"
    st.session_state.current_player = "O"  # Switch to AI
    
    winner = Helper.check_winner(st.session_state.board)
    if winner:
        update_game_state(winner)
    else:
        # AI plays
        st.session_state.board = Helper.ai_best_move(st.session_state.board, difficulty)
        winner = Helper.check_winner(st.session_state.board)
        if winner:
            update_game_state(winner)

def update_game_state(winner):
    if winner == "X":
        st.session_state.winner = "X"
        st.session_state.player_wins += 1
    elif winner == "O":
        st.session_state.winner = "O"
        st.session_state.ai_wins += 1
    elif winner == "Draw":
        st.session_state.winner = "Draw"
        st.session_state.draws += 1

def display_result():
    if st.session_state.winner == "X":
        st.success("🎉 You win! Congratulations! 🏆")
    elif st.session_state.winner == "O":
        st.error("🤖 AI wins! Better luck next time! 💻")
    else:
        st.info("🤝 It's a draw! Great minds think alike! 🧠")

def reset_game():
    st.session_state.board = Helper.initialize_board()
    st.session_state.current_player = "X"
    st.session_state.winner = None

if __name__ == "__main__":
    main()