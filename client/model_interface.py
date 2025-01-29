import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import cv2
import time
import rerun as rr

class TwistPredictor(ABC):
    """Abstract base class for all twist prediction models"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.model = self._create_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Standard image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Sliding window for near-zero detection
        self.reset_history()
    
    def reset_history(self):
        """Reset the sliding window history"""
        self.magnitude_history = []
        self.last_prediction_time = None
    
    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the underlying PyTorch model"""
        pass
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Predict twist commands from a frame
        Args:
            frame: BGR image as numpy array (from OpenCV)
        Returns:
            Dictionary with 'x' and 'theta' commands
        """
        pass
    
    def _log_window_stats(self, window_duration: float, magnitudes: List[float], 
                         all_near_zero: bool, cmd: Dict[str, float]):
        """Log window stats and commands to Rerun"""
        # Log window stats
        rr.log("window/duration", rr.Scalar(window_duration))
        rr.log("window/sample_count", rr.Scalar(len(magnitudes)))
        rr.log("window/all_near_zero", rr.Scalar(float(all_near_zero)))
        
        # Log all magnitudes in the window
        for i, mag in enumerate(magnitudes):
            rr.log(f"window/magnitude_{i}", rr.Scalar(mag))
        
        # Log the current command
        rr.log("command/x", rr.Scalar(cmd['x']))
        rr.log("command/theta", rr.Scalar(cmd['theta']))
        
        # Log whether we're in near-zero mode
        rr.log("status/near_zero_mode", rr.Scalar(float(all_near_zero and window_duration >= 0.15)))
    
    def _process_prediction(self, pred_numpy: np.ndarray, epsilon: float = 0.15) -> Dict[str, float]:
        """Common post-processing for predictions with sliding window for near-zero behavior"""
        # Calculate magnitude of the prediction
        magnitude = np.sqrt(pred_numpy[0]**2 + pred_numpy[2]**2)
        
        # Update timing
        current_time = time.time()
        if self.last_prediction_time is None:
            self.last_prediction_time = current_time
            dt = 0.05  # Assume initial dt of 50ms
        else:
            dt = current_time - self.last_prediction_time
            self.last_prediction_time = current_time
        
        # Update magnitude history
        self.magnitude_history.append((magnitude, dt))
        
        # Remove old entries until we're within the window
        window_duration = sum(dt for _, dt in self.magnitude_history)
        while window_duration > 0.25 and len(self.magnitude_history) > 1:  # Keep at least one sample
            _, removed_dt = self.magnitude_history.pop(0)
            window_duration -= removed_dt
        
        # Check if all magnitudes in window are below epsilon
        all_near_zero = all(mag < epsilon for mag, _ in self.magnitude_history)
        window_full = window_duration >= 0.15  # Need at least 0.15 seconds of history
        
        # Get list of just magnitudes for logging
        magnitudes = [mag for mag, _ in self.magnitude_history]
        
        # Debug prints
        print(f"Window stats: duration={window_duration:.3f}s, samples={len(self.magnitude_history)}, "
              f"all_near_zero={all_near_zero}, magnitudes={[f'{m:.3f}' for m in magnitudes]}")
        
        # Determine command based on conditions
        if all_near_zero and window_full:
            print("Near-zero behavior triggered!")
            cmd = {
                'x': float(pred_numpy[0]),  # Small forward velocity
                'theta': 5.0 * pred_numpy[2]  # No rotation when uncertain
            }
        else:
            cmd = {
                'x': float(pred_numpy[0]),
                'theta': float(pred_numpy[2])  # Skip y since we don't use it
            }
        
        # Log all stats to Rerun
        self._log_window_stats(window_duration, magnitudes, all_near_zero, cmd)
        
        return cmd

class CNNPredictor(TwistPredictor):
    """ResNet18-based single frame predictor"""
    
    def _create_model(self) -> nn.Module:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        return model
    
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        # Convert BGR to RGB and to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Transform and predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            pred_numpy = prediction.squeeze().cpu().numpy()
            return self._process_prediction(pred_numpy)

class LSTMPredictor(TwistPredictor):
    """LSTM-based sequence predictor"""
    
    def __init__(self, model_path: str, hidden_size: int = 128, num_layers: int = 1, 
                 device: Optional[torch.device] = None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_state = None  # Will store (h, c)
        super().__init__(model_path, device)
    
    def _create_model(self) -> nn.Module:
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        class CNNLSTMModel(nn.Module):
            def __init__(self, feature_extractor, hidden_size, num_layers):
                super().__init__()
                self.feature_extractor = feature_extractor
                self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 3)
            
            def forward(self, x, h_in=None):
                batch_size, seq_len, C, H, W = x.shape
                x = x.view(batch_size * seq_len, C, H, W)
                feats = self.feature_extractor(x)
                feats = feats.view(batch_size, seq_len, 512)
                
                if h_in is not None:
                    outputs, (h_out, c_out) = self.lstm(feats, h_in)
                else:
                    outputs, (h_out, c_out) = self.lstm(feats)
                
                last_output = outputs[:, -1, :]
                twist = self.fc(last_output)
                return twist, (h_out, c_out)
        
        return CNNLSTMModel(feature_extractor, self.hidden_size, self.num_layers)
    
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        # Convert BGR to RGB and to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Transform and add sequence dimension
        input_tensor = self.transform(image).unsqueeze(0).unsqueeze(0).to(self.device)
        # shape: [batch=1, seq_len=1, C, H, W]
        
        with torch.no_grad():
            prediction, self.hidden_state = self.model(input_tensor, self.hidden_state)
            pred_numpy = prediction.squeeze().cpu().numpy()
            return self._process_prediction(pred_numpy)
    
    def reset_state(self):
        """Reset the LSTM hidden state"""
        self.hidden_state = None

def create_predictor(model_type: str, model_path: str, **kwargs) -> TwistPredictor:
    """
    Factory function to create a predictor
    Args:
        model_type: 'cnn' or 'lstm'
        model_path: Path to the .pth model file
        **kwargs: Additional arguments for specific model types
    """
    if model_type.lower() == 'cnn':
        return CNNPredictor(model_path, **kwargs)
    elif model_type.lower() == 'lstm':
        return LSTMPredictor(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 