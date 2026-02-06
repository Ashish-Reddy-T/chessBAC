# Chess-ViT: Vision Transformer for Chess Move Prediction

## 1. Project Overview

Implements a **Vision Transformer (ViT)** to play chess. Unlike traditional engines (Stockfish) that use Minimax/AlphaBeta search, or early Deep Learning approaches (AlphaZero) that rely on ResNets + MCTS, this project treats chess board evaluation as a **Visual Pattern Recognition** problem.

I model the chessboard as an 8x8 grid of tokens, process it with Self-Attention mechanisms, and predict the most likely next move based on a dataset of Master-level games.

## 2. Directory Structure

```
BAC2/
├── data/
│   ├── raw/
│   │   └── games.csv           # Original Kaggle Dataset
│   └── processed/
│       ├── train_dataset.pt    # PyTorch Tensors for training
│       └── vocab.json          # Move <-> Integer mappings
├── src/
│   ├── config.py               # Hyperparameters (Layers, Heads, Embed Dim)
│   ├── dataset.py              # Parsing PGN/CSV -> 8x8 tensors
│   ├── model.py                # ViT Architecture (Patch Embeddings, Encoder)
│   ├── train.py                # Training Loop (Loss, Opt, Checkpointing)
│   └── play.py                 # Interface to play against the bot
├── checkpoints/                # Saved model weights
├── README.md
└── requirements.txt
```

## 3. Technical Architecture

### A. Input Representation (The "Vision" Part)

Instead of pixels, our "image" is the logic state of the board.

- **Grid:** 8x8 squares.
- **Tokens:** Each square is flattened into a token sequence of length 64.
- **Embeddings:**
  - `PieceEmbedding`: Maps piece type (P, N, B, R, Q, K, Empty) to vector $d_{model}$.
  - `PositionalEmbedding`: Learnable vector added to each square to retain grid geometry (A1 vs H8).

### B. The Core (Transformer Encoder)

- **Backbone:** Standard Transformer Encoder Stack.
- **Attention:** Multi-Head Self Attention (MSA) allows the model to "attend" to distant threats (e.g., a Bishop on a1 staring at h8) instantly, without the receptive field ramp-up of CNNs.
- **Layers:** 4-6 Layers (Scaled for 1-week turnaround + L4 GPU).
- **Heads:** 8 Heads.

### C. The Output (Policy Head)

- **Action Space:** Discrete classification task.
- **Classes:** We map every possible unique valid move in chess (e.g., `e2e4`, `a7a8q`) to an index.
- **Output:** Softmax distribution over all possible moves.

## 4. Implementation Plan

### Phase 1: Data Pipeline (Days 1-2)

- [ ] Load `games.csv`.
- [ ] Use `python-chess` to replay moves and extract `(Board State, Next Move)` pairs.
- [ ] Build a tokenizer for moves (UCI format) and pieces.
- [ ] Save processed tensors for efficient loading on HPC.

### Phase 2: Model Architecture (Day 3)

- [ ] Implement `ChessViT` class in PyTorch.
- [ ] Verify forward pass dimensions.
- [ ] Implement Masking (sanity check: ensure model doesn't cheat).

### Phase 3: Training (Day 4-5)

- [ ] Setup training loop on HPC (L4/A100).
- [ ] Train for ~10-20 epochs.
- [ ] Monitor CrossEntropyLoss and Top-1 / Top-3 Accuracy.

### Phase 4: Validating & "The Wow Factor" (Day 6-7)

- [ ] **Attention Maps:** Visualize _what_ the bot is looking at. (Does it stare at the King? Does it see the Knight fork?).
- [ ] **Playable Interface:** A simple Python script to play against it interactively.

## 5. References

1.  **AlphaZero**: _Mastering Chess and Shogi by Self-Play_ (Silver et al., 2017).
2.  **ViT**: _An Image is Worth 16x16 Words_ (Dosovitskiy et al., 2020).
3.  **Maiya Chess**: Modeling human probability rather than optimal play.
