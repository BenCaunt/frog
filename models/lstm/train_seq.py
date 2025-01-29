import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torch.nn.parallel import DataParallel
import torch.cuda.amp as amp

from datetime import datetime
from multi_seq_dataset import MultiSequenceRobotDataset
from tqdm import tqdm

class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # remove final FC
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        """
        x shape: [batch, seq_len, 3, 224, 224]
        returns: [batch, 3]
        """
        batch_size, seq_len, C, H, W = x.shape
        # Flatten for CNN
        x = x.view(batch_size * seq_len, C, H, W)  # [B*seq_len, 3, 224, 224]
        feats = self.feature_extractor(x)          # [B*seq_len, 512, 1, 1]
        feats = feats.view(batch_size, seq_len, 512)

        # LSTM
        outputs, (h_n, c_n) = self.lstm(feats)     # [batch, seq_len, hidden_size]
        # Last time step
        last_output = outputs[:, -1, :]           # [batch, hidden_size]
        twist = self.fc(last_output)              # [batch, 3]
        return twist

def get_device(force_cpu=False):
    """
    Determine the best available device with priority: CUDA > MPS > CPU
    """
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc='Training')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:  # Using mixed precision
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # Regular training
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc='Validating')
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str, help="Parent directory containing multiple subdirs of data.")
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable automatic mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()

    # Device setup
    device = get_device(force_cpu=args.force_cpu)
    print(f"Training on device: {device}")
    
    if device.type == "cuda":
        print(f"GPU(s) available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        if args.mixed_precision:
            print("Using automatic mixed precision training")

    # Initialize mixed precision scaler if needed
    scaler = amp.GradScaler() if device.type == "cuda" and args.mixed_precision else None

    # Build datasets with pinned memory for faster GPU transfer
    pin_memory = device.type == "cuda"
    
    # Training dataset
    train_dataset = MultiSequenceRobotDataset(
        parent_dir=args.parent_dir,
        seq_len=args.seq_len,
        validation=False
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    print(f"Loaded training dataset from '{args.parent_dir}' with {len(train_dataset)} total sequences")
    
    # Validation dataset
    try:
        val_dataset = MultiSequenceRobotDataset(
            parent_dir=args.parent_dir,
            seq_len=args.seq_len,
            validation=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        )
        print(f"Loaded validation dataset from '{args.parent_dir}' with {len(val_dataset)} total sequences")
        has_validation = True
    except ValueError:
        print("No validation data found, training without validation")
        has_validation = False

    # Build model
    model = CNNLSTMModel(hidden_size=args.hidden_size, num_layers=args.num_layers)
    
    # Multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
    model = model.to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    try:
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Training phase
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation phase
            if has_validation:
                val_loss = validate(model, val_loader, criterion, device)
                print(f"Validation Loss: {val_loss:.4f}")
                current_loss = val_loss
            else:
                current_loss = train_loss

            # Save best model based on validation loss (or training loss if no validation)
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"multi_seq_model_{timestamp}_best.pth"
                # Save the model state dict (handle DataParallel if used)
                torch.save(
                    model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                    out_name
                )
                print(f"Saved best model to {out_name}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"multi_seq_model_{timestamp}_final.pth"
    torch.save(
        model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
        out_name
    )
    print(f"Saved final model to {out_name}")

if __name__ == "__main__":
    main()