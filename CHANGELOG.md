# Changelog

## [v1.0.0] - 2026-02-06

### Added

- **Hybrid Inference Engine:** Implemented a new `get_best_move` logic in `src/play.py` that combines Neural Network intuition with hard-coded tactical rules.
  - **Tactical Override:** Instantly plays moves that capture major pieces (Rook/Queen) if available.
  - **Safety Filter:** Rejects candidate moves where a high-value piece moves to a square attacked by a lower-value piece (e.g., Queen to a Pawn-defended square).
- **Attention Visualization:** Created `src/visualize_attention.py` to generate heatmaps of the Transformer's focus, proving it attends to central board control.
- **HPC Infrastructure:** Added `hpc/` directory with `submit_train.slurm` and `setup_env.sh` for automating A100/L4 GPU training on NYU Greene/Burst clusters.

### Changed

- **Dataset Strategy:** Updated `src/dataset.py` to aggressively filter training data. Now only trains on games where both players have **ELO > 1500** (High Quality).
  - _Impact:_ Drastically improved model stability and reduced "blunder learning."
- **Model Architecture:** Finalized `ChessViT` architecture in `src/model.py` using Flattening (preserving spatial grid info) instead of Global Average Pooling.
- **Training Pipeline:** Updated `src/train.py` to support **WandB Logging** and **Auto-Resume** from checkpoints for robust Spot Instance training.

### Fixed

- **The "Free Queen" Bug:** Corrected an issue where the pure ViT model would ignore free material captures because it overly prioritized structural patterns. The new Hybrid Search fixes this.
- **Data Loading:** Fixed `dataset.py` execution block to properly check `if __name__ == "__main__"`, preventing accidental runs during import.

---

## [v0.5.0] - 2026-02-05

### Added

- Initial ViT implementation (`src/model.py`) with Piece and Positional Embeddings.
- PGN Parser (`src/dataset.py`) to convert raw games.csv into Tensor format.
- Basic Training Loop (`src/train.py`).
- Interactive CLI for playing against the bot (`src/play.py`).

### Changed

- Refactored project structure to separate source code (`src/`) from data and notebooks.
