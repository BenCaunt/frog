import pygame
import zenoh
import json
import time
import rerun as rr
import numpy as np
import cv2
import argparse
from teleop_constants import TELEOP_PUBLISH_RATE
from model_interface import LSTMPredictor, create_predictor

class Teleop:
    def __init__(self, model_type=None, model_path=None, **model_kwargs):
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
        self.predictor = None
        self.latest_frame = None
        self.model_enabled = False  # Track if model control is enabled
        self.prev_x_button = False  # For toggle detection
        if model_type and model_path:
            self.predictor = create_predictor(model_type, model_path, **model_kwargs)
            print(f"Initialized {model_type.upper()} predictor")
        
        # Try to find a joystick
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Found controller: {self.joystick.get_name()}")
        else:
            print("No controller found!")
            self.joystick = None
    
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
                    # Handle model toggle with X button (button 0 on PS5 controller)
                    x_button = self.joystick.get_button(0)
                    if x_button and not self.prev_x_button:  # Button just pressed
                        self.model_enabled = not self.model_enabled
                        state = "enabled" if self.model_enabled else "disabled"
                        print(f"Model control {state}")
                        if self.predictor:
                            # Reset model state when toggling
                            if isinstance(self.predictor, LSTMPredictor):
                                self.predictor.reset_state()
                            if not self.model_enabled:  # Only reset history when disabling
                                self.predictor.reset_history()
                    self.prev_x_button = x_button
                    
                    # Handle model disable with O button (button 1 on PS5 controller)
                    if self.joystick.get_button(1) and self.model_enabled:
                        self.model_enabled = False
                        print("Model control disabled")
                        if self.predictor:
                            if isinstance(self.predictor, LSTMPredictor):
                                self.predictor.reset_state()
                            self.predictor.reset_history()
                    
                    # Use model if enabled and available
                    use_model = self.model_enabled and self.predictor is not None and self.latest_frame is not None
                    
                    if use_model:
                        # Use model predictions
                        cmd = self.predictor.predict(self.latest_frame)
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
                        
                        # Only reset LSTM state during manual control, keep history
                        if self.predictor and isinstance(self.predictor, LSTMPredictor):
                            self.predictor.reset_state()
                
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
    parser.add_argument("--model-type", type=str, choices=['cnn', 'lstm'], help="Type of model to use")
    parser.add_argument("--model-path", type=str, help="Path to trained model (.pth file)")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for LSTM model")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers")
    
    args = parser.parse_args()
    
    # Only pass model kwargs if we're using a model
    model_kwargs = {}
    if args.model_type == 'lstm':
        model_kwargs.update({
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers
        })
    
    teleop = Teleop(
        model_type=args.model_type,
        model_path=args.model_path,
        **model_kwargs
    )
    teleop.run()

if __name__ == "__main__":
    main()
