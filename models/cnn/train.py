import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
from datetime import datetime

# -------------------------------------------------
# 1) Single-Directory Dataset
# -------------------------------------------------
class RobotDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: path to data/<SOME_NUMBER_BASED_ON_DATE_AND_TIME>
            e.g. data/2025-01-27_00-15-21

        Expected files:
          - metadata.json
          - sequences.json
          - images/ (contains frames)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")

        # Load metadata (optional usage)
        meta_path = os.path.join(root_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load sequences JSON
        seq_path = os.path.join(root_dir, "sequences.json")
        with open(seq_path, "r") as f:
            all_data = json.load(f)
            # all_data is a dict with key "sequences"
            self.sequences = all_data["sequences"]  # This is a list

        # Precompute full paths and aggregated labels
        self.samples = []
        for seq in self.sequences:
            # seq is a dict with keys: timestamp, image, twist_commands
            img_path = os.path.join(self.image_dir, seq["image"])

            # Extract or aggregate the twist commands (e.g., average them)
            if len(seq["twist_commands"]) > 0:
                x_vals = [tc["x"] for tc in seq["twist_commands"]]
                y_vals = [tc["y"] for tc in seq["twist_commands"]]
                t_vals = [tc["theta"] for tc in seq["twist_commands"]]
                avg_x = float(np.mean(x_vals))
                avg_y = float(np.mean(y_vals))
                avg_t = float(np.mean(t_vals))
            else:
                # If no twist_commands, default to zero
                avg_x, avg_y, avg_t = 0.0, 0.0, 0.0

            label = [avg_x, avg_y, avg_t]
            self.samples.append((img_path, label))

        # Simple transform: resize and normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float)
        return image, label_tensor


# -------------------------------------------------
# 2) Simple CNN Model for Regression
# -------------------------------------------------
class ImageToTwistModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: Use a pretrained ResNet18 and replace the final layer
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 3)  # output = (x, y, theta)

    def forward(self, x):
        return self.backbone(x)


# -------------------------------------------------
# 3) Helper: Build a "Combined" dataset from multiple subdirectories
# -------------------------------------------------
def build_combined_dataset(parent_dir):
    """
    parent_dir: e.g. "data"
      This function searches for all subdirectories within 'parent_dir' that contain
      'sequences.json' and 'images/' folder, then creates a RobotDataset for each
      and concatenates them into one big dataset.
    """
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d))]

    datasets = []
    for subdir in subdirs:
        seq_path = os.path.join(subdir, "sequences.json")
        img_dir = os.path.join(subdir, "images")
        if os.path.exists(seq_path) and os.path.exists(img_dir):
            # It's a valid dataset
            ds = RobotDataset(subdir)
            datasets.append(ds)
        else:
            print(f"Skipping {subdir}: no sequences.json or images/ folder found")

    if len(datasets) == 0:
        raise ValueError(f"No valid subdirectories found in {parent_dir}")

    # ConcatDataset merges them into one
    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


# -------------------------------------------------
# 4) Main training loop
# -------------------------------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a CNN model on robot data')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory. '
                        'This can be a single dataset folder (with sequences.json) '
                        'or a parent folder containing multiple subfolders.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--penalize-zero', action='store_true',
                        help='If set, add an extra loss term to penalize near-zero twists')
    parser.add_argument('--zero-penalty-weight', type=float, default=0.01,
                        help='Weight of the near-zero twist penalty')
    args = parser.parse_args()

    # Check if data_dir contains a sequences.json itself. If not, we assume it's a parent folder
    seq_path = os.path.join(args.data_dir, "sequences.json")
    img_dir  = os.path.join(args.data_dir, "images")
    if os.path.exists(seq_path) and os.path.exists(img_dir):
        # It's a single dataset
        print(f"Found a valid dataset in {args.data_dir}")
        dataset = RobotDataset(args.data_dir)
        dataset_name = os.path.basename(args.data_dir)
    else:
        # Possibly a parent directory
        print(f"No direct sequences.json found. Assuming '{args.data_dir}' is parent folder.")
        dataset = build_combined_dataset(args.data_dir)
        dataset_name = os.path.basename(os.path.normpath(args.data_dir)) + "_combined"

    # Check device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Instantiate model, loss, optimizer
    model = ImageToTwistModel().to(device)
    criterion = nn.MSELoss()  # or nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n----------- Training Configuration -----------")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Penalize near-zero twists: {args.penalize_zero}")
    if args.penalize_zero:
        print(f"Zero penalty weight: {args.zero_penalty_weight}")
    print("---------------------------------------------\n")

    # Train for specified epochs
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Base MSE or SmoothL1
            loss = criterion(outputs, labels)

            # Optionally add near-zero twist penalty:
            if args.penalize_zero:
                # We want to penalize small norms of predicted twist.
                # Example: penalty = alpha * mean( exp( -beta * ||u|| ) ), or something similar
                # But let's keep it simpler: alpha * mean(1 / (1 + ||u||^2)).
                # You can tune the exact function and weight.
                norm_sq = torch.sum(outputs**2, dim=1)  # shape: (batch,)
                penalty = torch.mean(1.0 / (1.0 + norm_sq))
                loss = loss + args.zero_penalty_weight * penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    # Save model with timestamp and dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"robot_twist_model_{dataset_name}_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
