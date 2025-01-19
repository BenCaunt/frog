#!/usr/bin/env python3
import time
import Adafruit_PCA9685
import argparse
from constants import MOTOR_PORTS, MOTOR_REVERSED, SERVO_MIN, SERVO_MAX, SERVO_NEUTRAL

class MotorDebug:
    def __init__(self, address=0x40, busnum=1):
        self.pwm = Adafruit_PCA9685.PCA9685(address=address, busnum=busnum)
        self.pwm.set_pwm_freq(60)
        self.motor_ports = MOTOR_PORTS
        self.motor_reversed = MOTOR_REVERSED
        
    def set_motor(self, port, speed):
        """
        Set motor speed (-1.0 to 1.0)
        """
        # Get motor name from port
        motor_name = next((name for name, p in self.motor_ports.items() if p == port), f"Unknown Port {port}")
        
        # Reverse speed if motor is configured as reversed
        for name, motor_port in self.motor_ports.items():
            if motor_port == port and self.motor_reversed[name]:
                speed = -speed
        
        # Convert -1.0 to 1.0 range to servo pulse range
        pulse = int(SERVO_NEUTRAL + (speed * (SERVO_MAX - SERVO_MIN) / 2))
        pulse = max(SERVO_MIN, min(SERVO_MAX, pulse))
        self.pwm.set_pwm(port, 0, pulse)
        print(f"Set motor {motor_name} (port {port}) to pulse {pulse}")

    def debug_sequence(self):
        """
        Run through each motor one at a time
        """
        TEST_SPEED = 0.3  # Use a moderate speed for testing
        
        try:
            while True:
                for motor_name, port in self.motor_ports.items():
                    print(f"\nTesting {motor_name} (Port {port})")
                    print("Running forward for 2 seconds...")
                    self.set_motor(port, TEST_SPEED)
                    time.sleep(2)
                    
                    print("Stopping for 1 second...")
                    self.set_motor(port, 0)
                    time.sleep(1)
                    
                    print("Running reverse for 2 seconds...")
                    self.set_motor(port, -TEST_SPEED)
                    time.sleep(2)
                    
                    print("Stopping...")
                    self.set_motor(port, 0)
                    time.sleep(1)
                    
                print("\nStarting sequence over...\n")
                
        except KeyboardInterrupt:
            print("\nStopping all motors...")
            for port in self.motor_ports.values():
                self.set_motor(port, 0)

    def forward_only_sequence(self):
        """
        Run all motors forward simultaneously
        """
        TEST_SPEED = 0.3  # Use a moderate speed for testing
        
        try:
            print("\nRunning all motors forward. Press CTRL+C to stop...")
            for motor_name, port in self.motor_ports.items():
                print(f"Starting {motor_name} (Port {port})")
                self.set_motor(port, TEST_SPEED)
            
            while True:
                time.sleep(0.1)  # Keep running until interrupted
                
        except KeyboardInterrupt:
            print("\nStopping all motors...")
            for port in self.motor_ports.values():
                self.set_motor(port, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Motor debug utility')
    parser.add_argument('--forward-only', action='store_true', help='Run all motors forward simultaneously')
    args = parser.parse_args()

    print("Starting motor debug sequence...")
    print("Press CTRL+C to stop")
    debugger = MotorDebug()
    
    if args.forward_only:
        debugger.forward_only_sequence()
    else:
        debugger.debug_sequence() 