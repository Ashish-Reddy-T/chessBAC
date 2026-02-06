
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import tqdm
import glob
import wandb
from model import ChessViT
import sys

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(data_path="data/processed"):
    print("Loading data...")
    # Load vocab
    with open(os.path.join(data_path, "vocab.json"), "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Load training tensors
    raw_data = torch.load(os.path.join(data_path, "train_data.pt"))
    
    inputs = torch.stack([item[0] for item in raw_data])
    targets = torch.tensor([item[1] for item in raw_data], dtype=torch.long)
    
    dataset = TensorDataset(inputs, targets)
    return dataset, vocab_size

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest .pt file in the directory based on modification time or name."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoints:
        return None
    # Sort by EPOCH number in filename (assumes format chess_vit_epoch_N.pt)
    try:
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return latest
    except:
        return max(checkpoints, key=os.path.getmtime)

def train(epochs=20, batch_size=128, learning_rate=3e-4, checkpoint_dir="checkpoints"):
    
    # Init WandB
    wandb.init(project="chess-vit-bac2", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "ViT-Flatten"
    })
    
    # 1. Prepare Data
    train_dataset, vocab_size = load_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Training Samples: {len(train_dataset)}")
    
    # 2. Init Model
    model = ChessViT(vocab_size=vocab_size).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    
    # 3. Resume from Checkpoint (Spot Instance Handling)
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming at Epoch {start_epoch+1}")
    else:
        print("Starting training from scratch...")

    model.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for boards, moves in progress_bar:
            boards, moves = boards.to(device), moves.to(device)
            
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == moves).sum().item()
            total += moves.size(0)
            
            current_loss = loss.item()
            current_acc = correct/total
            progress_bar.set_postfix({'loss': current_loss, 'acc': current_acc})
            
            # Log batch metrics
            wandb.log({"batch_loss": current_loss, "batch_acc": current_acc})
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Complete. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Log Epoch metrics
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "train_acc": epoch_acc})
        
        # Save Checkpoint (with optimizer state for resuming)
        checkpoint_path = os.path.join(checkpoint_dir, f"chess_vit_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

    wandb.finish()

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    # Scaled up batch size for A100/L4
    train(epochs=20, batch_size=256)
