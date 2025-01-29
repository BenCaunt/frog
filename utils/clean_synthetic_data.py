#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def clean_synthetic_data(data_dir: str = "data"):
    """
    Removes all folders under the data directory that contain '_synthetic' in their name.
    
    Args:
        data_dir (str): Path to the data directory relative to the project root
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_dir} directory not found")
        return
    
    synthetic_dirs = [d for d in data_path.iterdir() if d.is_dir() and "_synthetic" in d.name]
    
    if not synthetic_dirs:
        print("No synthetic data directories found")
        return
    
    print(f"Found {len(synthetic_dirs)} synthetic data directories")
    for dir_path in synthetic_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"Removed: {dir_path}")
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    clean_synthetic_data() 