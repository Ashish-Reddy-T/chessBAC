
import torch
import torch.nn.functional as F
import chess
import json
import os
import random
import pandas as pd
from model import ChessViT
from dataset import ChessDataset

class ChessBot:
    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = device
        
        # Load Vocab
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        
        # Initialize Model
        vocab_size = len(self.vocab)
        self.model = ChessViT(vocab_size=vocab_size).to(device)
        
        # Load Weights
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both full checkpoint dicts and direct state dicts
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Helper for board conversion
        self.dataset_helper = ChessDataset(pd.DataFrame({'moves': []}))

    def get_best_move(self, board, temperature=1.0):
        """
        Predicts the best legal move for the current board state.
        
        Args:
            board (chess.Board): Current board object.
            temperature (float): Controls randomness (high = more random).
            
        Returns:
            chess.Move: The selected best move.
        """
        # 1. Prepare Input
        board_tensor = self.dataset_helper.board_to_tensor(board).unsqueeze(0).to(self.device)
        
        # 2. Model Inference
        with torch.no_grad():
            logits = self.model(board_tensor)
        
        # 3. Get Legal Moves
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [m.uci() for m in legal_moves]
        
        # 4. Filter & MaskLogits
        # Create a tensor of -inf for illegal moves
        masked_logits = torch.full_like(logits, float('-inf'))
        
        valid_indices = []
        for move in legal_moves:
            uci = move.uci()
            if uci in self.vocab:
                idx = self.vocab[uci]
                masked_logits[0, idx] = logits[0, idx]
                valid_indices.append(idx)
            else:
                # If move is legal but NOT in our vocab (rare, but possible if training data missed it),
                # we can't predict it. 
                pass
        
        # Safety Check: If NO legal moves are in vocab (very rare), pick random legal move
        if len(valid_indices) == 0:
            print("Warning: No legal moves found in vocabulary! Picking random.")
            return random.choice(legal_moves)
            
        # 5. Softmax & Sampling
        probs = F.softmax(masked_logits / temperature, dim=1)
        
        # Greedy: Pick max probability
        # best_idx = torch.argmax(probs).item()
        
        # Sampling: Better for variety and avoiding loops
        best_idx = torch.multinomial(probs, 1).item()
        
        best_move_uci = self.vocab_inv[best_idx]
        return chess.Move.from_uci(best_move_uci)

def play_game():
    """
    Interactive Console Game: Human vs Bot
    """
    import pandas as pd # Needed for the dummy init in ChessBot
    
    # Path setup
    # Path setup
    MODEL_PATH = "checkpoints/chess_vit_epoch_20.pt" # Update this after training!
    VOCAB_PATH = "data/processed/vocab.json"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first!")
        return

    bot = ChessBot(MODEL_PATH, VOCAB_PATH)
    board = chess.Board()
    
    print("Welcome to Chess-ViT!")
    print("You are White. Enter moves in UCI format (e.g., e2e4).")
    print(board)
    
    while not board.is_game_over():
        # Human Move
        if board.turn == chess.WHITE:
            while True:
                try:
                    move_str = input("\nYour Move: ")
                    move = board.push_san(move_str)
                    break
                except ValueError:
                    print("Invalid move. Try again.")
        else:
            # Bot Move
            print("\nBot is thinking...")
            move = bot.get_best_move(board)
            board.push(move)
            print(f"Bot played: {move.uci()}")
            
        print("\n" + str(board))
        
    print(f"Game Over! Result: {board.result()}")

if __name__ == "__main__":
    play_game()
