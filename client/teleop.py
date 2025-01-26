import pygame
import zenoh
import json
import time
import rerun as rr
import numpy as np
import cv2

class Teleop:
    def __init__(self):
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
                    # Get joystick values
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
                    print(f"Published: {cmd}")
                
                time.sleep(0.02)  # 50Hz update rate
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Clean up
            if self.joystick:
                self.joystick.quit()
            pygame.quit()
            self.session.close()
            rr.disconnect()

if __name__ == "__main__":
    teleop = Teleop()
    teleop.run()
