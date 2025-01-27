# train_seq_multi.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from datetime import datetime
from multi_seq_dataset import MultiSequenceRobotDataset  # <-- put the dataset class here or inline
from tqdm import tqdm

# 1) Our CNN+LSTM model
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

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(device)  # [B, seq_len, 3, 224, 224]
        labels = labels.to(device)  # [B, 3]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str, help="Parent directory containing multiple subdirs of data.")
    parser.add_argument('--seq-len', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # Build dataset
    dataset = MultiSequenceRobotDataset(parent_dir=args.parent_dir, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Loaded dataset from '{args.parent_dir}' with {len(dataset)} total sequences")

    # Build model, optimizer, etc.
    model = CNNLSTMModel(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(args.epochs):
        loss_val = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss={loss_val:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"multi_seq_model_{timestamp}.pth"
    torch.save(model.state_dict(), out_name)
    print(f"Saved model to {out_name}")

if __name__ == "__main__":
    main()
