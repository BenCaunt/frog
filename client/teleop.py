import pygame
import zenoh
import json
import time
import rerun as rr
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import argparse
from teleop_constants import TELEOP_PUBLISH_RATE

class ImageToTwistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.backbone(x)

class Teleop:
    def __init__(self, model_path=None):
        pygame.init()
        pygame.joystick.init()
        
        # Initialize Rerun
        rr.init("RobotTeleop", spawn=True)
        
        # Initialize Zenoh
        self.session = zenoh.open(zenoh.Config())
        self.publisher = self.session.declare_publisher('robot/cmd')
        
        # Subscribe to camera feed
        self.subscriber = self.session.declare_subscriber(
            'robot/camera/frame',
            self._on_camera_frame
        )
        
        # Controller settings
        self.deadband = 0.1  # Ignore small inputs
        self.max_linear = 1.0  # Maximum forward/backward speed
        self.max_angular = 0.8  # Maximum rotation speed
        
        # Model setup
        self.model = None
        self.transform = None
        self.device = None
        self.latest_frame = None
        if model_path:
            self._setup_model(model_path)
        
        # Try to find a joystick
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Found controller: {self.joystick.get_name()}")
        else:
            print("No controller found!")
            self.joystick = None
    
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
    
    def _predict_twist(self, frame: np.ndarray) -> dict:
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
            pred_numpy = prediction.squeeze().cpu().numpy()
                        
            return {
                'x': float(pred_numpy[0]),
                'theta': float(pred_numpy[2])  # Skip y since we don't use it
            }
    
    def _on_camera_frame(self, sample):
        """Callback for camera frames"""
        try:
            # Convert Zenoh bytes to numpy array
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_frame = frame  # Store the latest frame for model inference
                # Convert BGR to RGB for rerun
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("camera", rr.Image(frame_rgb))
        except Exception as e:
            print(f"Error processing camera frame: {e}")
    
    def apply_deadband(self, value):
        if abs(value) < self.deadband:
            return 0
        return value
    
    def run(self):
        try:
            while True:
                pygame.event.pump()
                
                if self.joystick:
                    # Check if X button is pressed (button 0 on PS5 controller)
                    use_model = self.joystick.get_button(0) and self.model is not None and self.latest_frame is not None
                    
                    if use_model:
                        # Use model predictions
                        cmd = self._predict_twist(self.latest_frame)
                        if cmd:
                            self.publisher.put(json.dumps(cmd))
                            print(f"Model command: {cmd}")
                    else:
                        # Manual teleop
                        forward = -self.apply_deadband(self.joystick.get_axis(1))  # Y axis
                        rotation = -self.apply_deadband(self.joystick.get_axis(2))  # Right X axis
                        
                        # Scale to max speeds
                        forward *= self.max_linear
                        rotation *= self.max_angular    
                        
                        # Create command message
                        cmd = {
                            'x': forward,
                            'theta': rotation
                        }
                        
                        # Publish command
                        self.publisher.put(json.dumps(cmd))
                        print(f"Teleop command: {cmd}")
                
                time.sleep(1 / TELEOP_PUBLISH_RATE)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Clean up
            if self.joystick:
                self.joystick.quit()
            pygame.quit()
            self.session.close()
            rr.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Teleoperate the robot with optional model control")
    parser.add_argument("--model", type=str, help="Optional path to trained model (.pth file)")
    args = parser.parse_args()
    
    teleop = Teleop(model_path=args.model)
    teleop.run()

if __name__ == "__main__":
    main()
