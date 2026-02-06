
import pandas as pd
import chess
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
import json
import os

class ChessDataset(Dataset):
    def __init__(self, games_df, move_vocab=None):
        """
        Args:
            games_df (pd.DataFrame): DataFrame containing 'moves' column
            move_vocab (dict): Dictionary mapping UCI moves to integers. 
                               If None, it will be built from the data.
        """
        self.games = games_df['moves'].tolist()
        self.move_vocab = move_vocab if move_vocab else {}
        self.data = [] # List of (board_tensor, move_index) tuples
        
        # Piece encoding: 
        # White: P=0, N=1, B=2, R=3, Q=4, K=5
        # Black: p=6, n=7, b=8, r=9, q=10, k=11
        # Empty: 12
        self.piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
            None: 12
        }

    def build_vocab(self):
        """
        Iterates through all games to build a comprehensive Move Vocabulary.
        Only needed if we don't have a pre-computed vocab.
        """
        unique_moves = set()
        print("Building Move Vocabulary...")
        for move_seq in tqdm.tqdm(self.games):
            board = chess.Board()
            moves = move_seq.split()
            for move_san in moves:
                try:
                    move = board.push_san(move_san)
                    unique_moves.add(move.uci())
                except ValueError:
                    # Invalid move string in dataset
                    break
        
        # Create mapping
        self.move_vocab = {move: i for i, move in enumerate(sorted(unique_moves))}
        return self.move_vocab

    def board_to_tensor(self, board):
        """
        Converts a chess.Board object into a 8x8 integer tensor.
        Each square holds the piece ID (0-12).
        """
        board_tensor = torch.zeros(64, dtype=torch.long)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                symbol = piece.symbol()
                board_tensor[i] = self.piece_map[symbol]
            else:
                board_tensor[i] = self.piece_map[None]
        
        return board_tensor.view(8, 8)

    def process_games(self, max_games=None):
        """
        Parses games and stores them as (Board, Move) pairs in memory.
        """
        print("Processing games into tensors...")
        count = 0
        
        games_to_process = self.games[:max_games] if max_games else self.games
        
        for move_seq in tqdm.tqdm(games_to_process):
            board = chess.Board()
            moves = move_seq.split()
            
            for move_san in moves:
                # 1. Capture current state
                state = self.board_to_tensor(board)
                
                try:
                    # 2. Get the move that WAS played
                    move = board.push_san(move_san)
                    move_uci = move.uci()
                    
                    if move_uci in self.move_vocab:
                        move_idx = self.move_vocab[move_uci]
                        self.data.append((state, move_idx))
                except ValueError:
                    break
            
            count += 1
            
        print(f"Processed {len(self.data)} positions from {count} games.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def save_processed_data(df, output_dir="data/processed", max_games=1000):
    """
    Driver function to process data and save tensors to disk.
    We limit max_games initially to test the pipeline.
    """
    dataset = ChessDataset(df)
    
    # 1. Build Vocab first (usually you'd do this on the whole dataset)
    move_vocab = dataset.build_vocab()
    
    # Save vocab
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(move_vocab, f)
        
    # 2. Process Data
    dataset.process_games(max_games=max_games)
    
    # 3. Save as PyTorch object (much faster to load than re-parsing)
    torch.save(dataset.data, os.path.join(output_dir, "train_data.pt"))
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    df = pd.read_csv("data/raw/games.csv")
    
    # Filter for quality: Only keep games where BOTH players are decent (>1500 ELO)
    # This prevents the model from learning "bad habits" from beginners.
    print(f"Original games: {len(df)}")
    df = df[ (df['white_rating'] >= 1500) & (df['black_rating'] >= 1500) ]
    print(f"Filtered High-Quality games: {len(df)}")
    
    # Process full dataset
    save_processed_data(df, max_games=None)
