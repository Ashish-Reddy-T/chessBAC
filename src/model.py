
import torch
import torch.nn as nn
import math

class ChessViT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Embeddings
        # 13 types of pieces (6 white, 6 black, 1 empty)
        self.piece_embedding = nn.Embedding(13, embed_dim)
        
        # Positional Embedding: 64 squares
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Policy Head
        # Flattened input: 64 squares * embed_dim
        self.policy_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 64),
            nn.Linear(embed_dim * 64, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Good initialization is crucial for Transformers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: (Batch, 8, 8) Board Tensor
        """
        B, H, W = x.shape
        
        # Flatten board to (Batch, 64)
        x = x.view(B, -1)
        
        # Token Embeddings: (Batch, 64) -> (Batch, 64, Dim)
        x = self.piece_embedding(x)
        
        # Add Position Embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer Pass
        # Output: (Batch, 64, Dim)
        x = self.transformer_encoder(x)
        
        # Flatten for Policy Head
        # (Batch, 64, Dim) -> (Batch, 64 * Dim)
        x = x.view(B, -1)
        
        # Policy Head -> Logits over moves
        # (Batch, Vocab_Size)
        logits = self.policy_head(x)
        
        return logits

def test_model():
    """Simple test to verify forward pass"""
    vocab_size = 4000 # Approximation
    model = ChessViT(vocab_size=vocab_size)
    
    # Dummy board (Batch=2, 8x8)
    dummy_input = torch.randint(0, 13, (2, 8, 8))
    
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.shape == (2, vocab_size)
    print("ChessViT Forward Pass Successful!")

if __name__ == "__main__":
    test_model()
