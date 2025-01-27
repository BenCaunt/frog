import cv2
import zenoh
from zenoh import Config
import time

from constants import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_FRAME_KEY,
    FLIP_FRAME,
    CROP_TO_MONO,
    CROP_TO_RIGHT_FRAME
)

def crop_frame_to_mono(frame):
    """
    Crops a stereo frame to a single camera view
    Args:
        frame: Input stereo frame
    Returns:
        Cropped monocular frame
    """
    # Calculate the midpoint - stereo frames are side by side
    mid_x = frame.shape[1] // 2
    
    if CROP_TO_RIGHT_FRAME:
        # Take the right half of the frame
        return frame[:, mid_x:]
    else:
        # Take the left half of the frame
        return frame[:, :mid_x]

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

                # Crop to mono if enabled
                if CROP_TO_MONO:
                    frame = crop_frame_to_mono(frame)

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