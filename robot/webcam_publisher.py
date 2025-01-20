import cv2
import zenoh
from zenoh import Config
import time

from constants import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_FRAME_KEY,
    FLIP_FRAME
)



def main():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Force MJPG format for higher FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize Zenoh session with config
    with zenoh.open(Config()) as z_session:
        print("Press Ctrl+C to quit.")
        
        # For FPS calculation
        frame_count = 0
        fps_start_time = time.monotonic()
        
        try:
            while True:
                frame_start_time = time.monotonic()
                
                # Capture frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip frame if enabled
                if FLIP_FRAME:
                    frame = cv2.flip(frame, -1)  # -1 flips both horizontally and vertically

                # Publish to Zenoh
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    z_session.put(CAMERA_FRAME_KEY, buffer.tobytes())

                # Calculate and maintain FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Print FPS every 30 frames
                    current_time = time.monotonic()
                    fps = frame_count / (current_time - fps_start_time)
                    print(f"FPS: {fps:.1f}")
                    frame_count = 0
                    fps_start_time = current_time

                # Small sleep to maintain consistent frame rate
                frame_end_time = time.monotonic()
                frame_duration = frame_end_time - frame_start_time
                sleep_time = max(0, 1.0/CAMERA_FPS - frame_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cap.release()

if __name__ == "__main__":
    main() 