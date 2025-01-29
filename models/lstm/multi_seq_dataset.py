import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class MultiSequenceRobotDataset(Dataset):
    """
    Gathers time-contiguous frame sequences from multiple subdirectories under a parent folder.

    Parent folder structure (example):
       data/
         2025-01-27_00-15-21_real/
            images/
            sequences.json
         2025-01-27_00-30-10_real/
            images/
            sequences.json
         validate_2025-01-27_00-45-30_real/
            images/
            sequences.json
         ...
    Each subdirectory is treated separately: we do NOT assume continuity across subdirs.
    Directories starting with 'validate_' are used for validation only.
    """
    def __init__(self, 
                 parent_dir: str,
                 seq_len: int = 5,
                 validation: bool = False):
        """
        parent_dir: Path to the parent data folder (e.g., "data/").
                    We'll scan all immediate subdirectories for "sequences.json" & "images/".
        seq_len: number of consecutive frames in a sequence.
        validation: if True, only use directories starting with 'validate_',
                   if False, exclude those directories.

        We store the data in:
           self.subdir_frames[i] = a list of (timestamp, img_full_path, [avg_x, avg_y, avg_t]) sorted by timestamp
           self.indices = a list of (subdir_idx, start_idx) for each valid sequence
        """
        super().__init__()
        self.parent_dir = parent_dir
        self.seq_len = seq_len
        self.validation = validation
        self.subdir_frames = []  # list of lists
        self.indices = []        # global list of (subdir_idx, start_idx)

        # 1) Discover subdirectories
        subdirs = [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]
        subdirs.sort()  # optional, just to have a consistent order

        # Filter based on validation flag
        subdirs = [d for d in subdirs if 
                  (d.startswith(os.path.join(parent_dir, "validate_")) == validation)]

        # 2) For each subdirectory, parse the frames
        subdir_count = 0
        for subdir_path in subdirs:
            seq_path = os.path.join(subdir_path, "sequences.json")
            img_dir = os.path.join(subdir_path, "images")

            if os.path.exists(seq_path) and os.path.exists(img_dir):
                frames = self._load_subdir(subdir_path)
                if len(frames) >= self.seq_len:
                    # store frames
                    self.subdir_frames.append(frames)
                    # add global indices for each possible sequence start
                    n_seq = len(frames) - self.seq_len + 1
                    for start_idx in range(n_seq):
                        self.indices.append((subdir_count, start_idx))
                    subdir_count += 1
                else:
                    print(f"Skipping {subdir_path}: not enough frames for seq_len={self.seq_len}")
            else:
                print(f"Skipping {subdir_path}: missing sequences.json or images/ folder")

        if len(self.indices) == 0:
            raise ValueError(f"No valid {'validation' if validation else 'training'} subdirectories found under {parent_dir} with seq_len={self.seq_len}")

        # 3) Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_subdir(self, subdir_path):
        """
        Parse a single subdir's sequences.json, returns list of
        (timestamp, img_full_path, [avg_x, avg_y, avg_theta]),
        sorted by timestamp.
        """
        seq_path = os.path.join(subdir_path, "sequences.json")
        img_dir = os.path.join(subdir_path, "images")

        with open(seq_path, "r") as f:
            data = json.load(f)
            seqs = data["sequences"]

        # gather frames
        frames = []
        for entry in seqs:
            ts = entry["timestamp"]
            img_name = entry["image"]
            twist_cmds = entry["twist_commands"]
            if len(twist_cmds) > 0:
                x_vals = [tc["x"] for tc in twist_cmds]
                y_vals = [tc["y"] for tc in twist_cmds]
                t_vals = [tc["theta"] for tc in twist_cmds]
                avg_x = float(np.mean(x_vals))
                avg_y = float(np.mean(y_vals))
                avg_t = float(np.mean(t_vals))
            else:
                avg_x, avg_y, avg_t = 0.0, 0.0, 0.0

            full_img_path = os.path.join(img_dir, img_name)
            frames.append((ts, full_img_path, [avg_x, avg_y, avg_t]))

        # sort by timestamp
        frames.sort(key=lambda x: x[0])
        return frames

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1) Identify which subdir and start index
        subdir_idx, start_idx = self.indices[idx]
        frames = self.subdir_frames[subdir_idx]

        # 2) Collect seq_len frames: [start_idx, ..., start_idx + seq_len - 1]
        seq_frames = frames[start_idx : start_idx + self.seq_len]

        # 3) Load images
        img_tensors = []
        for (ts, img_path, label_) in seq_frames:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            img_tensors.append(img)

        # 4) The label is from the *last* frame in the sequence
        last_label = seq_frames[-1][2]  # [avg_x, avg_y, avg_t]
        label_tensor = torch.tensor(last_label, dtype=torch.float)

        # 5) Stack image tensors: shape = [seq_len, 3, 224, 224]
        images = torch.stack(img_tensors, dim=0)

        return images, label_tensor
