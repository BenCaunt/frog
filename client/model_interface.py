import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import cv2

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
    
    def _process_prediction(self, pred_numpy: np.ndarray, epsilon: float = 0.05) -> Dict[str, float]:
        """Common post-processing for predictions"""
        # Check if prediction magnitude is near zero
        magnitude = np.sqrt(pred_numpy[0]**2 + pred_numpy[2]**2)
        
        if magnitude < epsilon:
            # If near zero, move forward slowly
            return {
                'x': 0.15,  # Small forward velocity
                'theta': 0.0  # No rotation
            }
        else:
            return {
                'x': float(pred_numpy[0]),
                'theta': float(pred_numpy[2])  # Skip y since we don't use it
            }

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