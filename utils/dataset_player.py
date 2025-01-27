import argparse
import json
import time
from pathlib import Path
import cv2
import rerun as rr
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional

class ImageToTwistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.backbone(x)

class DatasetPlayer:
    def __init__(self, dataset_path: Path, model_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        # Load metadata and sequences
        self.metadata = self._load_json("metadata.json")
        self.sequences = self._load_json("sequences.json")["sequences"]
        
        # Initialize rerun
        rr.init("Dataset Player", spawn=True)
        
        # Setup model if provided
        self.model = None
        self.transform = None
        self.device = None
        if model_path:
            self._setup_model(model_path)
        
    def _setup_model(self, model_path: str):
        """Initialize the model for inference"""
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.model = ImageToTwistModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
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
    
    def _predict_twist(self, frame: np.ndarray) -> np.ndarray:
        """Predict twist commands from an image using the loaded model"""
        if self.model is None:
            return None
            
        # Convert BGR to RGB and to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Transform and predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            return prediction.squeeze().cpu().numpy()
    
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
                    
                    # Get model prediction if available
                    if self.model is not None:
                        predicted_twist = self._predict_twist(frame)
                        rr.log("predicted_twist/x", rr.Scalar(predicted_twist[0]))
                        rr.log("predicted_twist/y", rr.Scalar(predicted_twist[1]))
                        rr.log("predicted_twist/theta", rr.Scalar(predicted_twist[2]))
                
                # Plot twist commands up to current time
                current_time = sequence["timestamp"]
                while (twist_idx < len(twist_timeseries) and 
                       twist_timeseries[twist_idx][0] <= current_time):
                    t, values = twist_timeseries[twist_idx]
                    # Log each component as a Scalar
                    rr.log("actual_twist/x", rr.Scalar(values[0]))
                    rr.log("actual_twist/y", rr.Scalar(values[1]))
                    rr.log("actual_twist/theta", rr.Scalar(values[2]))
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
    parser.add_argument("--model", type=str, help="Optional path to trained model (.pth file)")
    
    args = parser.parse_args()
    
    try:
        player = DatasetPlayer(args.dataset_path, args.model)
        player.play(speed=args.speed)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 