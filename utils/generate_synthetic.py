#!/usr/bin/env python3

import os
import shutil
import json
import argparse
from PIL import Image
from torchvision import transforms
import torch
import random
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time

def process_image_task(args):
    """Process a single image with augmentation"""
    real_img_path, synth_img_path, seed = args
    
    # Set seeds for reproducibility within this process
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Define augmentation pipeline
    augmentation = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1, contrast=0.05, saturation=0.0, hue=0.0
        ),
        transforms.RandomErasing(p=0.25),
        transforms.RandomRotation(degrees=0.5),
    ])
    
    try:
        with Image.open(real_img_path).convert("RGB") as pil_img:
            img_tensor = transforms.ToTensor()(pil_img)
            aug_tensor = augmentation(img_tensor)
            aug_pil = transforms.ToPILImage()(aug_tensor)
            aug_pil.save(synth_img_path)
        return (synth_img_path, True)
    except Exception as e:
        print(f"Error processing {real_img_path}: {str(e)}")
        return (synth_img_path, False)

def prepare_dataset_tasks(real_subdir_path, parent_dir, synthetic_index, base_seed):
    """Prepare all tasks for a single synthetic dataset"""
    real_subdir = os.path.basename(real_subdir_path)
    synth_subdir_name = real_subdir.replace("_real", f"_synthetic_{synthetic_index+1}")
    synth_subdir_path = os.path.join(parent_dir, synth_subdir_name)
    
    # Create directory structure
    synth_images_dir = os.path.join(synth_subdir_path, "images")
    os.makedirs(synth_images_dir, exist_ok=True)
    
    # Copy metadata and sequences files
    real_metadata = os.path.join(real_subdir_path, "metadata.json")
    synth_metadata = os.path.join(synth_subdir_path, "metadata.json")
    if os.path.exists(real_metadata):
        shutil.copy(real_metadata, synth_metadata)
    
    sequences_json = os.path.join(real_subdir_path, "sequences.json")
    synth_sequences = os.path.join(synth_subdir_path, "sequences.json")
    shutil.copy(sequences_json, synth_sequences)
    
    # Prepare image processing tasks
    images_dir = os.path.join(real_subdir_path, "images")
    all_images = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
    ]
    
    tasks = []
    for img_name in all_images:
        real_img_path = os.path.join(images_dir, img_name)
        synth_img_path = os.path.join(synth_images_dir, img_name)
        # Generate a unique seed for each image
        image_seed = base_seed + hash(img_name) + synthetic_index
        tasks.append((real_img_path, synth_img_path, image_seed))
    
    return synth_subdir_name, tasks

def generate_synthetic_data(parent_dir, num_synthetic=8, num_processes=None):
    """
    Generate synthetic datasets using a single process pool for all operations.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Discover real subdirs
    all_subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    real_subdirs = [d for d in all_subdirs if d.endswith('_real')]
    
    if not real_subdirs:
        print(f"No directories ending with '_real' found in {parent_dir}. Nothing to do.")
        return
    
    print(f"Found {len(real_subdirs)} '_real' dataset(s) to augment")
    
    # Prepare all tasks
    all_tasks = []
    dataset_task_counts = {}  # Keep track of tasks per dataset
    base_seed = int(time.time())
    
    for real_subdir in real_subdirs:
        real_subdir_path = os.path.join(parent_dir, real_subdir)
        
        # Verify required files exist
        images_dir = os.path.join(real_subdir_path, "images")
        sequences_json = os.path.join(real_subdir_path, "sequences.json")
        if not os.path.exists(images_dir) or not os.path.exists(sequences_json):
            print(f"Skipping {real_subdir_path} - missing images/ or sequences.json")
            continue
        
        # Prepare tasks for each synthetic copy
        for i in range(num_synthetic):
            dataset_name, tasks = prepare_dataset_tasks(real_subdir_path, parent_dir, i, base_seed)
            dataset_task_counts[dataset_name] = len(tasks)
            all_tasks.extend(tasks)
    
    # Process all images in parallel using a single pool
    results = {}
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(all_tasks), desc="Processing images") as pbar:
            for img_path, success in pool.imap_unordered(process_image_task, all_tasks):
                # Track results by dataset
                dataset_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                if dataset_name not in results:
                    results[dataset_name] = {"success": 0, "total": dataset_task_counts[dataset_name]}
                if success:
                    results[dataset_name]["success"] += 1
                pbar.update(1)
    
    # Print summary
    print("\nSynthetic Dataset Generation Summary:")
    for dataset_name, stats in sorted(results.items()):
        print(f" -> {dataset_name}: Successfully processed {stats['success']}/{stats['total']} images")
    
    print("\nAll synthetic datasets generated successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate N synthetic datasets (random color/warp) from each '_real' dataset using parallel processing."
    )
    parser.add_argument("parent_dir", type=str, help="Path to parent data directory")
    parser.add_argument("--n", type=int, default=8, help="Number of synthetic copies per dataset")
    parser.add_argument("--processes", type=int, default=None, 
                       help="Number of processes to use (default: number of CPU cores)")
    args = parser.parse_args()
    
    generate_synthetic_data(args.parent_dir, num_synthetic=args.n, num_processes=args.processes)

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on Windows
    mp.freeze_support()
    main()