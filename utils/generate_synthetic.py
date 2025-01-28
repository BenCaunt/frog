#!/usr/bin/env python3

import os
import shutil
import json
import argparse
from PIL import Image
from torchvision import transforms
import torch
import random

def generate_synthetic_data(parent_dir, num_synthetic=8):
    """
    1) Find all subdirectories under 'parent_dir' that end with '_real'
    2) For each such directory, create N new subdirectories, each named:
         original_subdir.replace('_real', f'_synthetic_{i}')
       (i = 1..num_synthetic)
    3) Copy metadata.json and sequences.json, but augment each frame in 'images/'.
    """

    # 1) Discover subdirs that end with '_real'
    all_subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    real_subdirs = [d for d in all_subdirs if d.endswith('_real')]

    if len(real_subdirs) == 0:
        print(f"No directories ending with '_real' found in {parent_dir}. Nothing to do.")
        return

    # 2) Define your augmentation pipeline(s).
    #    For example, random color jitter + perspective transform.
    #    You can tweak these values or add more (rotation, etc.).
    #    We'll apply the same pipeline each time, but you can randomize further if desired.
    augmentation = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    ])

    print(f"Found {len(real_subdirs)} '_real' dataset(s) to augment.")
    for real_subdir in real_subdirs:
        real_subdir_path = os.path.join(parent_dir, real_subdir)

        # Check for images/ and sequences.json
        images_dir = os.path.join(real_subdir_path, "images")
        sequences_json = os.path.join(real_subdir_path, "sequences.json")
        if not os.path.exists(images_dir) or not os.path.exists(sequences_json):
            print(f"Skipping {real_subdir_path} - missing images/ or sequences.json.")
            continue

        # 3) Generate each synthetic copy
        print(f"Generating {num_synthetic} synthetic dataset(s) for: {real_subdir}")
        for i in range(num_synthetic):
            synth_subdir_name = real_subdir.replace("_real", f"_synthetic_{i+1}")
            synth_subdir_path = os.path.join(parent_dir, synth_subdir_name)

            # (a) Create new directory structure
            synth_images_dir = os.path.join(synth_subdir_path, "images")
            os.makedirs(synth_images_dir, exist_ok=True)

            # (b) Copy metadata.json if present
            real_metadata = os.path.join(real_subdir_path, "metadata.json")
            synth_metadata = os.path.join(synth_subdir_path, "metadata.json")
            if os.path.exists(real_metadata):
                shutil.copy(real_metadata, synth_metadata)

            # (c) Copy sequences.json
            synth_sequences = os.path.join(synth_subdir_path, "sequences.json")
            shutil.copy(sequences_json, synth_sequences)

            # (d) Augment each image and save into the new images/ folder
            all_images = [
                f for f in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, f))
            ]
            for img_name in all_images:
                real_img_path = os.path.join(images_dir, img_name)
                synth_img_path = os.path.join(synth_images_dir, img_name)

                # Load
                with Image.open(real_img_path).convert("RGB") as pil_img:
                    # Convert to tensor for transforms
                    img_tensor = transforms.ToTensor()(pil_img)

                    # Apply augmentation
                    # (We do a small random seed shuffle to get varied results each dataset.)
                    # You can remove or adapt for deterministic runs.
                    seed = torch.seed()  # store the global seed
                    random.seed(seed)
                    torch.manual_seed(seed)

                    aug_tensor = augmentation(img_tensor)
                    # Convert back to PIL
                    aug_pil = transforms.ToPILImage()(aug_tensor)

                    # Save
                    aug_pil.save(synth_img_path)

            print(f" -> Created synthetic dataset: {synth_subdir_name}")

    print("All synthetic datasets generated successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate N synthetic datasets (random color/warp) from each '_real' dataset."
    )
    parser.add_argument("parent_dir", type=str, help="Path to parent data directory")
    parser.add_argument("--n", type=int, default=8, help="Number of synthetic copies per dataset")
    args = parser.parse_args()

    generate_synthetic_data(args.parent_dir, num_synthetic=args.n)

if __name__ == "__main__":
    main()
