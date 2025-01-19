#!/usr/bin/env python3
from drivebase import DriveBase
import time
import signal
import sys

def signal_handler(sig, frame):
    print('Shutting down...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting drivebase...")
    drivebase = DriveBase()
    
    # Keep the script running
    while True:
        time.sleep(1)
