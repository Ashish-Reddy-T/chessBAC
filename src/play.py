
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

    def get_best_move(self, board, temperature=0.8):
        """
        Hybrid Approach: ViT Intuition + 1-Ply Tactical Search
        """
        # 1. Prepare Input
        board_tensor = self.dataset_helper.board_to_tensor(board).unsqueeze(0).to(self.device)
        
        # 2. Model Inference
        with torch.no_grad():
            logits = self.model(board_tensor)
        
        # 3. Get Legal Moves
        legal_moves = list(board.legal_moves)
        
        # --- TACTICAL OVERRIDE ---
        # Before asking the model, check for obvious winning captures (1-Ply Search)
        # Value mapping: P=1, N=3, B=3.5, R=5, Q=9
        for move in legal_moves:
            if board.is_capture(move):
                # What are we capturing?
                if board.is_en_passant(move):
                    captured_piece_val = 1
                else:
                    captured_piece = board.piece_at(move.to_square)
                    if captured_piece:
                        val_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.5, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                        captured_piece_val = val_map.get(captured_piece.piece_type, 0)
                        
                        # If we capture a Queen or Rook free/favorably, DO IT.
                        # Simple rule: If capture value >= 5 (Rook/Queen), prioritize it.
                        if captured_piece_val >= 5:
                            # Verify it's safe? (Too complex for 1-ply, but usually capturing a queen is good)
                            print(f"Tactical Override: Capturing {captured_piece.symbol()} with {move.uci()}")
                            return move

        # 4. Filter & MaskLogits
        masked_logits = torch.full_like(logits, float('-inf'))
        valid_indices = []
        
        for move in legal_moves:
            uci = move.uci()
            if uci in self.vocab:
                idx = self.vocab[uci]
                masked_logits[0, idx] = logits[0, idx]
                valid_indices.append(idx)
        
        if not valid_indices:
            return random.choice(legal_moves)
            
        # 5. Sampling
        probs = F.softmax(masked_logits / temperature, dim=1)
        
        # Heuristic Safety Check: Don't just blunder pieces immediately
        # Sample TOP K instead of just 1, and pick the first SAFE one
        K = 5
        top_indices = torch.multinomial(probs, K, replacement=True)[0]
        
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

        for idx in top_indices:
            move_uci = self.vocab_inv[idx.item()]
            move = chess.Move.from_uci(move_uci)
            
            # Safety Check: Are we moving a valuable piece to a square attacked by a pawn/minor?
            if board.is_capture(move): 
                return move # Taking things is usually fun/safe-ish in 1-ply logic
                
            # Who is moving?
            my_piece = board.piece_at(move.from_square)
            if not my_piece: continue # Should not happen
            my_val = piece_values.get(my_piece.piece_type, 0)
            
            # Is the destination attacked by opponent?
            # board.attackers(color, square)
            opponent = not board.turn
            attackers = board.attackers(opponent, move.to_square)
            
            is_safe = True
            for attacker_sq in attackers:
                attacker_piece = board.piece_at(attacker_sq)
                atk_val = piece_values.get(attacker_piece.piece_type, 1)
                
                # If a pawn (1) attacks my Queen (9), this is BAD.
                if atk_val < my_val:
                    is_safe = False
                    break
            
            if is_safe:
                return move
        
        # If all top K moves are unsafe, just panic and play the first one (ViT's top choice)
        fallback_uci = self.vocab_inv[top_indices[0].item()]
        return chess.Move.from_uci(fallback_uci)

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
