# Chess-ViT: Vision Transformer for Chess Move Prediction

## 1. Project Overview

Implements a **Vision Transformer (ViT)** to play chess. Unlike traditional engines (Stockfish) that use Minimax/AlphaBeta search, or early Deep Learning approaches (AlphaZero) that rely on ResNets + MCTS, this project treats chess board evaluation as a **Visual Pattern Recognition** problem.

I model the chessboard as an 8x8 grid of tokens, process it with Self-Attention mechanisms, and predict the most likely next move based on a curated dataset of over 10,000 High-Quality Master-level games (>1500 ELO).

## 2. Directory Structure

```
BAC2/
├── data/
│   ├── raw/
│   │   └── games.csv           # Original Kaggle Dataset
│   ├── processed/
│   │   ├── train_dataset.pt    # PyTorch Tensors for training
│   │   └── vocab.json          # Move <-> Integer mappings
├── src/
│   ├── config.py               # Hyperparameters (Layers, Heads, Embed Dim)
│   ├── dataset.py              # Parsing PGN/CSV -> 8x8 tensors (Filtering >1500 ELO)
│   ├── model.py                # ViT Architecture (Patch Embeddings, Encoder)
│   ├── train.py                # Training Loop (Loss, Opt, Checkpointing, WandB)
│   ├── play.py                 # Hybrid Inference Engine (ViT + 1-Ply Search)
│   └── visualize_attention.py  # Generates attention heatmaps
├── hpc/                        # SLURM scripts for NYU Greene/Burst
├── checkpoints/                # Saved model weights
├── README.md
└── requirements.txt
```

## 3. Technical Architecture

### A. Input Representation (The "Vision" Part)

Instead of pixels, our "image" is the logical state of the board.

- **Grid:** 8x8 squares.
- **Tokens:** Each square is flattened into a token sequence of length 64.
- **Embeddings:**
  - `PieceEmbedding`: Maps piece type (P, N, B, R, Q, K, Empty) to vector $d_{model}$.
  - `PositionalEmbedding`: Learnable vector added to each square to retain grid geometry (A1 vs H8).

### B. The Core (Transformer Encoder)

- **Backbone:** Standard Transformer Encoder Stack.
- **Attention:** Multi-Head Self Attention (MSA) allows the model to "attend" to distant threats (e.g., a Bishop on a1 staring at h8) instantly.
- **Layers:** 6 Layers, 8 Heads.

### C. The Output (Policy Head)

- **Action Space:** Discrete classification task over ~1871 unique moves.
- **Output:** Softmax distribution over all possible moves.

### D. Hybrid Inference Engine (src/play.py)

To bridge the gap between "Intuition" and "Calculation," the final bot uses a hybrid approach:

- **Tactical Override:** Before checking the Neural Network, the bot scans for "Free Material" (e.g., capturing a hanging Queen) and takes it immediately.
- **Safety Filter:** The bot verifies the Neural Network's suggestion. If the ViT suggests moving a Queen to a square attacked by a Pawn, the bot rejects it and picks the next best move.
- **Result:** A bot that plays with "Human Intuition" (ViT) but has the "Common Sense" to avoid obvious one-move blunders.

## 4. Features Implemented

### Phase 1: Data Pipeline

- [x] Load `games.csv` and filter for High Quality (>1500 ELO).
- [x] Use `python-chess` to replay moves and extract `(Board State, Next Move)` pairs.
- [x] Build a tokenizer for moves (UCI format) and pieces.

### Phase 2: Model Architecture

- [x] Implement `ChessViT` class in PyTorch using Flattening strategy.
- [x] Verify forward pass dimensions.

### Phase 3: Training

- [x] Setup training loop on HPC (L4/A100).
- [x] Trained for 20 epochs, achieving **71.4% Accuracy**.
- [x] Integrated WandB for live loss monitoring.

### Phase 4: Validating

- [x] **Attention Maps:** Visualization script (`visualize_attention.py`) confirms the model focuses on Center Control (`e4/d4`).
- [x] **Playable Interface:** Interactive CLI (`play.py`) with Hybrid Search logic.

## 5. How to Run

### Setup

```bash
pip install -r requirements.txt
```

### 1. Generating Data (Optional - Data is already processed)

```bash
python src/dataset.py
```

### 2. Training (requires GPU)

```bash
python src/train.py
```

### 3. Playing Against the Bot

```bash
# Ensure checkpoints/chess_vit_epoch_20.pt exists
python src/play.py
```

### 4. Visualizing Attention

```bash
python src/visualize_attention.py
# outputs attention_map.png
```

## 6. References

1.  **AlphaZero**: _Mastering Chess and Shogi by Self-Play_ (Silver et al., 2017).
2.  **ViT**: _An Image is Worth 16x16 Words_ (Dosovitskiy et al., 2020).
3.  **Maia Chess**: Modeling human probability rather than optimal play.
