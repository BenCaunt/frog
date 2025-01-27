import zenoh
import json
import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import threading
from queue import Queue
import logging

from teleop_constants import TELEOP_PUBLISH_RATE

class DatasetLogger:
    def __init__(self):
        # Create dataset directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = Path("data") / timestamp
        self.images_dir = self.dataset_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DatasetLogger")
        
        # Setup Zenoh
        self.session = zenoh.open(zenoh.Config())
        
        # Initialize data structures
        self.frame_queue = Queue()
        self.twist_buffer = []
        self.last_frame_time = None
        self.sequence_data = {
            "sequences": []  # Will store image-twist sequences
        }
        
        # Threading lock for twist buffer
        self.twist_lock = threading.Lock()
        
        # Subscribe to camera and twist topics
        self.camera_sub = self.session.declare_subscriber(
            'robot/camera/frame',
            self._on_camera_frame
        )
        self.twist_sub = self.session.declare_subscriber(
            'robot/cmd',
            self._on_twist_command
        )
        
        self.logger.info(f"Logging dataset to: {self.dataset_dir}")
        
        # Save dataset metadata
        self.save_metadata()
    
    def save_metadata(self):
        """Save dataset metadata including timing and configuration"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "frame_rate": 30,  # Assumed from your constants
            "twist_rate": TELEOP_PUBLISH_RATE,  # Target rate for twist commands
            "robot_type": "tank_drive",
            "control_dims": ["x", "y", "theta"],  # Standard format for all robots
            "description": "Tank drive robot exploration dataset"
        }
        
        with open(self.dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _on_camera_frame(self, sample):
        """Handle incoming camera frames"""
        try:
            # Convert Zenoh bytes to numpy array
            np_arr = np.frombuffer(sample.payload.to_bytes(), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                current_time = time.time()
                
                # Save the frame
                frame_filename = f"frame_{current_time:.6f}.jpg"
                cv2.imwrite(str(self.images_dir / frame_filename), frame)
                
                # Get accumulated twist commands since last frame
                with self.twist_lock:
                    twist_commands = self.twist_buffer.copy()
                    self.twist_buffer = []
                
                # Create sequence entry
                sequence_entry = {
                    "timestamp": current_time,
                    "image": frame_filename,
                    "twist_commands": twist_commands
                }
                
                self.sequence_data["sequences"].append(sequence_entry)
                self.last_frame_time = current_time
                
                # Periodically save sequence data
                if len(self.sequence_data["sequences"]) % 100 == 0:
                    self.save_sequence_data()
                    self.logger.info(f"Saved {len(self.sequence_data['sequences'])} sequences")
                
        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")
    
    def _on_twist_command(self, sample):
        """Handle incoming twist commands"""
        try:
            data = json.loads(sample.payload.to_string())
            
            # Convert tank drive commands to standard twist format
            twist = {
                "timestamp": time.time(),
                "x": data.get('x', 0.0),
                "y": 0.0,  # Tank drive has no Y movement
                "theta": data.get('theta', 0.0)
            }
            
            with self.twist_lock:
                self.twist_buffer.append(twist)
                
        except Exception as e:
            self.logger.error(f"Error processing twist command: {e}")
    
    def save_sequence_data(self):
        """Save the current sequence data to disk"""
        sequence_file = self.dataset_dir / "sequences.json"
        with open(sequence_file, "w") as f:
            json.dump(self.sequence_data, f, indent=2)
    
    def cleanup(self):
        """Cleanup resources and save data"""
        try:
            # Save final sequence data
            self.save_sequence_data()
            
            # Close subscriptions explicitly
            if hasattr(self, 'camera_sub'):
                self.camera_sub.undeclare()
            if hasattr(self, 'twist_sub'):
                self.twist_sub.undeclare()
            
            # Close Zenoh session
            if hasattr(self, 'session'):
                self.session.close()
            
            self.logger.info("Dataset logging complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def run(self):
        """Main run loop"""
        try:
            self.logger.info("Dataset logger running. Press Ctrl+C to stop...")
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.cleanup()

if __name__ == "__main__":
    logger = DatasetLogger()
    logger.run() 