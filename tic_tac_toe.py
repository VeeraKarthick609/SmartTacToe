import streamlit as st
import torch
import numpy as np
from utils.helper import Helper
from rl_agent.agent import DQNAgent

def main():
    st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮", layout="centered")
    
    # Custom CSS for responsiveness
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        height: 100px;
        font-size: 2rem;
    }
    @media (max-width: 600px) {
        .stButton > button {
            height: 80px;
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎮 Tic-Tac-Toe Challenge 🤖")

    # AI type selection
    ai_type = st.radio("Select AI Type:", ["Rule-based AI", "DQN Agent"], key="ai_type")

    # Difficulty selection (only for rule-based AI)
    difficulty = None
    if ai_type == "Rule-based AI":
        difficulty = st.selectbox("Select AI Difficulty Level:", ["Easy", "Medium", "Hard"], key="difficulty")

    # Check if AI type or difficulty changed and reset game state if needed
    if "previous_ai_type" not in st.session_state:
        st.session_state.previous_ai_type = ai_type

    if "previous_difficulty" not in st.session_state:
        st.session_state.previous_difficulty = difficulty

    if (st.session_state.previous_ai_type != ai_type) or (st.session_state.previous_difficulty != difficulty):
        reset_game()
        st.session_state.previous_ai_type = ai_type
        st.session_state.previous_difficulty = difficulty

    # Initialize DQN Agent
    if "dqn_agent" not in st.session_state:
        st.session_state.dqn_agent = DQNAgent(device="cpu")
        st.session_state.dqn_agent.load_model("weights/dqn_final_model.pth")

    # Initialize session state
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
    
    # Create a 3x3 grid
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            with cols[j]:
                if st.session_state.board[i][j] == "X":
                    st.button("❌", key=f"{i}-{j}", disabled=True)
                elif st.session_state.board[i][j] == "O":
                    st.button("⭕", key=f"{i}-{j}", disabled=True)
                else:
                    if st.button("🔲", key=f"{i}-{j}") and st.session_state.winner is None:
                        make_move(i, j, ai_type, difficulty)
                        st.rerun()

    # Display winner or draw
    if st.session_state.winner:
        display_result()

    # Restart the game
    if st.button("🔄 New Game", key="restart"):
        reset_game()

def make_move(i, j, ai_type, difficulty):
    st.session_state.board[i][j] = "X"
    st.session_state.current_player = "O"  # Switch to AI
    
    winner = Helper.check_winner(st.session_state.board)
    if winner:
        update_game_state(winner)
    else:
        # AI plays
        if ai_type == "Rule-based AI":
            st.session_state.board = Helper.ai_best_move(st.session_state.board, difficulty)
        else:  # DQN Agent
            flattened_board = [item for sublist in st.session_state.board for item in sublist]
            available_actions = [i for i, x in enumerate(flattened_board) if x == " "]
            action = st.session_state.dqn_agent.choose_action(flattened_board, available_actions)
            row, col = action // 3, action % 3
            st.session_state.board[row][col] = "O"

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
