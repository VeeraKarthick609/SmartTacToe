# SmartTacToe

A simple tic-tac-toe game developed during my extra holidays due to rain. 🌧️

### Features
- **Rule-Based AI**: Utilizes the Minimax algorithm for optimal play.
- **Deep Q-Learning (DQN) Agent**: An AI model that learns to play tic-tac-toe through reinforcement learning.

Made for fun! 🎮

### Get Started

To set up and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/VeeraKarthick609/SmartTacToe
   cd SmartTacToe
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows
   ```bash
   venv\Scripts\activate
   ```
    - Linux/MacOS
    ```bash
    source venv/bin/activate
    ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Launch the Streamlit App

To start the game, run the following command:
```bash
streamlit run tic_tac_toe.py
```

### Train the Model

To train the DQN model, use the command:
```bash
python rl_agent/train.py
```

### Enjoy Playing!

Try your skills against our AI and have fun! Feel free to contribute or suggest improvements.