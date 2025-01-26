import argparse
import json
import time
from pathlib import Path
import cv2
import rerun as rr
import numpy as np
from typing import Dict, List

class DatasetPlayer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        # Load metadata and sequences
        self.metadata = self._load_json("metadata.json")
        self.sequences = self._load_json("sequences.json")["sequences"]
        
        # Initialize rerun
        rr.init("Dataset Player", spawn=True)
        
        # No need to pre-create timeseries plots - they'll be created automatically
        
    def _load_json(self, filename: str) -> Dict:
        """Load and parse a JSON file from the dataset directory"""
        file_path = self.dataset_path / filename
        if not file_path.exists():
            raise ValueError(f"Missing required file: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _get_twist_timeseries(self) -> List[tuple]:
        """Extract twist commands into a timeseries format"""
        timeseries_data = []
        
        for sequence in self.sequences:
            for twist in sequence["twist_commands"]:
                timestamp = twist["timestamp"]
                timeseries_data.append((
                    timestamp,
                    [twist["x"], twist["y"], twist["theta"]]
                ))
        
        return sorted(timeseries_data, key=lambda x: x[0])
    
    def play(self, speed: float = 1.0):
        """
        Play the dataset visualization
        
        Args:
            speed: Playback speed multiplier (1.0 = realtime)
        """
        print(f"Playing dataset from: {self.dataset_path}")
        print(f"Playback speed: {speed}x")
        
        # Get start time of first sequence
        start_time = self.sequences[0]["timestamp"]
        
        # Extract twist timeseries for plotting
        twist_timeseries = self._get_twist_timeseries()
        twist_idx = 0
        
        try:
            for sequence in self.sequences:
                # Load and display image
                image_path = self.dataset_path / "images" / sequence["image"]
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    # Convert BGR to RGB for rerun
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rr.log("camera", rr.Image(frame_rgb))
                
                # Plot twist commands up to current time
                current_time = sequence["timestamp"]
                while (twist_idx < len(twist_timeseries) and 
                       twist_timeseries[twist_idx][0] <= current_time):
                    t, values = twist_timeseries[twist_idx]
                    # Log each component as a Scalar
                    rr.log("twist/x", rr.Scalar(values[0]))
                    rr.log("twist/y", rr.Scalar(values[1]))
                    rr.log("twist/theta", rr.Scalar(values[2]))
                    twist_idx += 1
                
                # Calculate sleep time based on playback speed
                if sequence != self.sequences[-1]:
                    next_time = self.sequences[self.sequences.index(sequence) + 1]["timestamp"]
                    sleep_duration = (next_time - current_time) / speed
                    time.sleep(max(0, sleep_duration))
                
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")
        
        print("Playback complete")

def main():
    parser = argparse.ArgumentParser(description="Play back a robot dataset with visualization")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Playback speed multiplier (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        player = DatasetPlayer(args.dataset_path)
        player.play(speed=args.speed)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 