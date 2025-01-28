#!/usr/bin/env python3
import os
from pathlib import Path

def rename_datasets():
    """Rename dataset directories to append '_real' if not already present"""
    data_dir = Path("data")
    
    # Ensure data directory exists
    if not data_dir.exists():
        print("Data directory not found")
        return
    
    # Get all subdirectories in data/
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        # Skip if already has _real suffix
        if subdir.name.endswith("_real"):
            continue
            
        # Check if it's a dataset directory (has sequences.json and images/)
        if (subdir / "sequences.json").exists() and (subdir / "images").exists():
            new_name = subdir.with_name(f"{subdir.name}_real")
            
            # Rename the directory
            try:
                subdir.rename(new_name)
                print(f"Renamed: {subdir.name} -> {new_name.name}")
            except Exception as e:
                print(f"Error renaming {subdir.name}: {e}")
        else:
            print(f"Skipping {subdir.name}: not a dataset directory")

if __name__ == "__main__":
    rename_datasets() 