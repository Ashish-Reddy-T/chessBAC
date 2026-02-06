
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import chess
import json
import os
import numpy as np
from model import ChessViT
from dataset import ChessDataset

# Setup logic
MODEL_PATH = "checkpoints/chess_vit_epoch_20.pt"
VOCAB_PATH = "data/processed/vocab.json"
DEVICE = "cpu"

def load_model():
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    model = ChessViT(vocab_size=vocab_size).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, vocab

def capture_attention_weights(model, x):
    """
    Runs a forward pass and captures attention weights from the LAST transformer layer.
    """
    # 1. Embeddings
    B, H, W = x.shape
    x = x.view(B, -1)
    x = model.piece_embedding(x)
    x = x + model.pos_embedding
    
    # 2. Transformer Pass (Manual Unroll to get weights)
    # We only care about the last layer for "final decision" attention
    attentions = []
    
    for i, layer in enumerate(model.transformer_encoder.layers):
        # We need to hack the forward pass of the MultiheadAttention to get weights
        # Standard implementation of TransformerEncoderLayer:
        # x = norm1(x + self_attn(x, x, x)[0])
        # x = norm2(x + feed_forward(x))
        
        # Access the self attention module
        attn_module = layer.self_attn
        
        # Manual Forward of Self-Attention
        # x is (Batch, SeqLen, Dim) -> (Batch, 64, Dim)
        # MultiheadAttention expects (SeqLen, Batch, Dim) if batch_first=False
        # Our model defined batch_first=True
        
        # Run attention with average_attn_weights=False usually, inside it returns (output, weights)
        # But we need need_weights=True
        
        attn_output, attn_weights = attn_module(x, x, x, need_weights=True, average_attn_weights=True)
        # attn_weights: (Batch, TargetSeq, SourceSeq) -> (1, 64, 64)
        
        # Finish the layer arithmetic (Standard Transformer Add & Norm)
        x = x + layer.dropout1(attn_output)
        x = layer.norm1(x)
        
        x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = x + layer.dropout2(x2)
        x = layer.norm2(x)
        
        if i == len(model.transformer_encoder.layers) - 1:
            attentions.append(attn_weights)

    return attentions[-1] # Return last layer weights (1, 64, 64)

def visualize_board_attention(fen, save_path="attention_map.png"):
    model, vocab = load_model()
    dataset = ChessDataset(pd.DataFrame({'moves': []}))
    
    board = chess.Board(fen)
    tensor = dataset.board_to_tensor(board).unsqueeze(0).to(DEVICE) # (1, 8, 8)
    
    # Get Weights
    with torch.no_grad():
        # attn_matrix: (1, 64, 64) -> How much each square attends to every other square
        attn_matrix = capture_attention_weights(model, tensor)
    
    # We want to see: "Which squares are most important for the GLOBAL decision?"
    # A simple heuristic: Aggregate attention a square receives from everything else
    # Sum over columns: (1, 64) -> Total attention PAID TO this square
    # or Sum over rows: (1, 64) -> Total attention PAID BY this square
    
    # Let's visualize "Importance": How much attention does each square RECEIVE from the board?
    # This roughly highlights "Active Pieces"
    attention_score = attn_matrix[0].sum(dim=0).view(8, 8).numpy()
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Overlay on a chessboard pattern
    ax = sns.heatmap(attention_score, annot=True, fmt=".1f", cmap="viridis", square=True,
                     xticklabels=['a','b','c','d','e','f','g','h'],
                     yticklabels=['8','7','6','5','4','3','2','1'])
    
    plt.title("ChessViT Attention Heatmap (Where the Model Looks)\nDarker = More Focus")
    plt.savefig(save_path)
    print(f"Saved attention map to {save_path}")
    print("Board Order:")
    print(board)

if __name__ == "__main__":
    # Example: A sharp mid-game position
    # "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1" (Ruy Lopezish)
    fen = "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 4"
    visualize_board_attention(fen)
