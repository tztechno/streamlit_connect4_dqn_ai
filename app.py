
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import os
import streamlit as st

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Connect4AI:
    def __init__(self):
        self.ROWS = 6
        self.COLS = 7
        self.board_size = self.ROWS * self.COLS
        self.action_size = self.COLS
        
        # DQNの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.board_size, self.action_size).to(self.device)
        self.target_net = DQN(self.board_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
    
        # ハイパーパラメータ
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        
    def get_state(self, board):
        return torch.FloatTensor(board.flatten()).to(self.device)
    
    def get_action(self, state, valid_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            
        # 無効な手に対してはペナルティを与える
        valid_q_values = q_values.clone()
        for i in range(self.COLS):
            if i not in valid_moves:
                valid_q_values[i] = float('-inf')
                
        return valid_q_values.argmax().item()
    
    def get_valid_moves(self, board):
        return [col for col in range(self.COLS) if board[0][col] == 0]
    
    def make_move(self, board, col, player):
        # Create a deep copy of the board to avoid modifying the original
        new_board = np.copy(board)
        
        # Find the lowest empty row in the selected column
        for row in range(self.ROWS-1, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                return new_board, row
                
        # If column is full, return original board and -1 to indicate invalid move
        return board, -1
    
    def check_winner(self, board, player):
        # 横方向
        for row in range(self.ROWS):
            for col in range(self.COLS-3):
                if all(board[row][col+i] == player for i in range(4)):
                    return True
        
        # 縦方向
        for row in range(self.ROWS-3):
            for col in range(self.COLS):
                if all(board[row+i][col] == player for i in range(4)):
                    return True
        
        # 右斜め上方向
        for row in range(self.ROWS-3):
            for col in range(self.COLS-3):
                if all(board[row+i][col+i] == player for i in range(4)):
                    return True
        
        # 左斜め上方向
        for row in range(self.ROWS-3):
            for col in range(3, self.COLS):
                if all(board[row+i][col-i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_draw(self, board):
        return len(self.get_valid_moves(board)) == 0
    
    def get_reward(self, board, player):
        if self.check_winner(board, player):
            return 1.0
        elif self.check_winner(board, -player):
            return -1.0
        elif self.is_draw(board):
            return 0.0
        return None
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            board = np.zeros((self.ROWS, self.COLS))
            done = False
            player = 1
            
            while not done:
                state = self.get_state(board)
                valid_moves = self.get_valid_moves(board)
                
                if player == 1:  # AIの手番
                    action = self.get_action(state, valid_moves)
                    next_board, _ = self.make_move(board, action, player)
                    reward = self.get_reward(next_board, player)
                    
                    if reward is not None:
                        done = True
                    else:
                        reward = 0.0
                    
                    next_state = self.get_state(next_board)
                    self.memory.append((state, action, reward, next_state, done))
                    
                    if len(self.memory) >= self.batch_size:
                        self._train_step()
                    
                    board = next_board
                else:  # 相手の手番（ランダム）
                    action = random.choice(valid_moves)
                    board, _ = self.make_move(board, action, player)
                    
                    if self.get_reward(board, player) is not None:
                        done = True
                
                player = -player
            
            # εの減衰
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Target networkの更新
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if (episode + 1) % 1000 == 0:
                print(f"Episode: {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.3f}")
    
    def _train_step(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, save_dir='models'):
        """Save the trained model and optimizer states"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save models and training state
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)  # Convert deque to list for saving
        }, os.path.join(save_dir, 'connect4_model.pth'))
        
        print(f"Model saved to {os.path.join(save_dir, 'connect4_model.pth')}")
    
    def load_model(self, model_path):
        """Load the trained model and optimizer states, handling both local files and URLs."""
        
        if model_path.startswith("http://") or model_path.startswith("https://"):
            # It's a URL, download it
            try:
                import requests
                response = requests.get(model_path, stream=True)
                response.raise_for_status()
    
                filename = os.path.basename(model_path)
                filepath = filename  # Saves in the current directory. Change as needed.
    
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
    
                model_path = filepath  # Update model_path to the local file path
    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error downloading model from {model_path}: {e}")
            except Exception as e:
                raise Exception(f"Error saving model: {e}")
    
        # Now, model_path should be a local file path, whether it was originally a URL or not.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
    
        # Load the saved state
        checkpoint = torch.load(model_path, map_location=self.device)
    
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=10000)
    
        print(f"Model loaded from {model_path}")

# Streamlit app
def main():
    st.title("Connect 4 AI Game")
    st.write("Play Connect 4 against a reinforcement learning AI!")
    
    # Move training options to the sidebar
    with st.sidebar:
        st.header("AI Training Options")
        training_episodes = st.number_input("Training Episodes", min_value=100, max_value=50000, value=5000, step=100)
        if st.button("Train AI"):
            with st.spinner(f"Training AI for {training_episodes} episodes..."):
                st.session_state.game.train(num_episodes=training_episodes)
                st.session_state.game.save_model()
            st.success("Training complete and model saved!")
    
    # Initialize session state to store game state
    if 'board' not in st.session_state:
        st.session_state.board = np.zeros((6, 7))
    if 'game' not in st.session_state:
        st.session_state.game = Connect4AI()
        # Load the model if it exists
        try:
            st.session_state.game.load_model('models/connect4_model.pth')
            st.success("Loaded pre-trained AI model")
        except FileNotFoundError:
            st.warning("No pre-trained model found. Using untrained AI.")
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'result_message' not in st.session_state:
        st.session_state.result_message = ""
    if 'current_player' not in st.session_state:
        st.session_state.current_player = 1  # 1 for human, -1 for AI
    if 'human_player' not in st.session_state:
        st.session_state.human_player = 1  # 1 if human is player 1, -1 if human is player 2
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False  # Track if game has started
    
    # Options for starting a new game - Always show these options unless a game is in progress
    if not st.session_state.game_started or st.session_state.game_over:
        st.header("Who goes first?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("I'll Go First (X)"):
                reset_game(human_first=True)
                st.session_state.game_started = True
                st.rerun()
        with col2:
            if st.button("AI Goes First (O)"):
                reset_game(human_first=False)
                st.session_state.game_started = True
                st.rerun()
    
    # Display the game board if a game has started
    if st.session_state.game_started:
        st.write("### Game Board")
        draw_board()
        
        # Display whose turn it is
        if not st.session_state.game_over:
            if st.session_state.current_player == st.session_state.human_player:
                st.write("### Your Turn")
            else:
                st.write("### AI's Turn")
                # If it's AI's turn, make a move
                make_ai_move()
                st.rerun()  # Force a rerun to update the board after AI move
        else:
            st.write(f"### {st.session_state.result_message}")
            if st.button("Start New Game"):
                st.session_state.game_started = False
                reset_game()
                st.rerun()
        
        # Column buttons for human player
        if not st.session_state.game_over and st.session_state.current_player == st.session_state.human_player:
            valid_moves = st.session_state.game.get_valid_moves(st.session_state.board)
            cols = st.columns(7)
            for col in range(7):
                with cols[col]:
                    # Check if this is a valid move
                    button_disabled = col not in valid_moves
                    if st.button(f"{col}", key=f"col_{col}", disabled=button_disabled):
                        make_human_move(col)

def draw_board():
    """Draw the Connect 4 board using Streamlit"""
    board = st.session_state.board
    
    # CSS の定義（7×7 のグリッド）
    cell_style = """
    <style>
    .board-container {
        display: grid;
        grid-template-columns: repeat(7, 50px);
        grid-template-rows: repeat(7, 50px);
        gap: 5px;
        justify-content: center;
    }
    .board-cell {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .empty-cell { background-color: #e0e0e0; }
    .player1-cell { background-color: #ff6b6b; color: white; }
    .player2-cell { background-color: #4ecdc4; color: white; }
    .turn-cell {
        background-color: #ddd;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """
    
    # CSS 適用
    st.markdown(cell_style, unsafe_allow_html=True)
    
    # グリッドコンテナ作成
    board_html = '<div class="board-container">'
    
    # 6×7 のゲーム盤面を作成
    for row in range(6):
        for col in range(7):
            cell_value = board[row][col]
            if cell_value == 0:
                cell_class = "empty-cell"
                symbol = ""
            elif cell_value == 1:
                cell_class = "player1-cell"
                symbol = "X"
            else:
                cell_class = "player2-cell"
                symbol = "O"
            
            board_html += f'<div class="board-cell {cell_class}">{symbol}</div>'
    
    # 7行目（Your Turn 表示用の行）
    for col in range(7):
        board_html += f'<div class="board-cell turn-cell">Your Turn</div>'
    
    board_html += '</div>'
    
    # Streamlit に表示
    st.markdown(board_html, unsafe_allow_html=True)


def make_human_move(col):
    """Make a human move in the specified column"""
    game = st.session_state.game
    board = st.session_state.board
    human_player = st.session_state.human_player
    
    # Make the move
    new_board, _ = game.make_move(board, col, human_player)
    st.session_state.board = new_board
    
    # Check for game over
    check_game_over(human_player)
    
    # Switch players if game is not over
    if not st.session_state.game_over:
        st.session_state.current_player = -st.session_state.human_player
        st.rerun()

def make_ai_move():
    """Make an AI move"""
    game = st.session_state.game
    board = st.session_state.board
    ai_player = -st.session_state.human_player
    
    # Get valid moves
    valid_moves = game.get_valid_moves(board)
    if not valid_moves:
        # If no valid moves, it's a draw
        st.session_state.game_over = True
        st.session_state.result_message = "It's a draw! 🤝"
        return
    
    # Get AI action - ensure we get a valid move
    state = game.get_state(board)
    action = game.get_action(state, valid_moves, training=False)
    
    # Double-check that action is valid
    if action not in valid_moves:
        # If somehow we got an invalid move, just pick a random valid one
        action = random.choice(valid_moves)
    
    # Make the move
    new_board, _ = game.make_move(board, action, ai_player)
    st.session_state.board = new_board
    
    # Check for game over
    check_game_over(ai_player)
    
    # Switch players if game is not over
    if not st.session_state.game_over:
        st.session_state.current_player = st.session_state.human_player

def check_game_over(player):
    """Check if the game is over after a move by the specified player"""
    game = st.session_state.game
    board = st.session_state.board
    
    # Check for win
    if game.check_winner(board, player):
        st.session_state.game_over = True
        if player == st.session_state.human_player:
            st.session_state.result_message = "You win! 🎉"
        else:
            st.session_state.result_message = "AI wins! 🤖"
    
    # Check for draw
    elif game.is_draw(board):
        st.session_state.game_over = True
        st.session_state.result_message = "It's a draw! 🤝"

def reset_game(human_first=True):
    """Reset the game state"""
    st.session_state.board = np.zeros((6, 7))
    st.session_state.game_over = False
    st.session_state.result_message = ""
    st.session_state.human_player = 1 if human_first else -1
    st.session_state.current_player = 1  # Player 1 always goes first

if __name__ == "__main__":
    main()

