#!/usr/bin/env python3
import subprocess
import signal
import sys
import time

def main():
    print("Starting robot components...")
    
    # Start processes
    processes = []
    try:
        # Start drivebase
        print("Starting drivebase...")
        drivebase = subprocess.Popen(["python", "robot/drivebase.py"])
        processes.append(drivebase)
        
        # Start webcam publisher
        print("Starting webcam publisher...")
        webcam = subprocess.Popen(["python", "robot/webcam_publisher.py"])
        processes.append(webcam)
        
        print("\nRobot is running! Press Ctrl+C to stop.")
        
        # Monitor processes
        while all(p.poll() is None for p in processes):
            time.sleep(0.1)
            
        # If we get here, a process died
        for p in processes:
            if p.poll() is not None:
                print(f"Process {p.args} died with code {p.returncode}")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Terminate all processes
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Process {p.args} failed to terminate, killing...")
                    p.kill()
        
        print("All processes stopped.")

if __name__ == "__main__":
    main()
