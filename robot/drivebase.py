from __future__ import division
import time
import zenoh
import Adafruit_PCA9685
import json

class DriveBase:
    def __init__(self, address=0x40, busnum=1):
        self.pwm = Adafruit_PCA9685.PCA9685(address=address, busnum=busnum)
        self.pwm.set_pwm_freq(60)
        
        # Motor configuration
        self.SERVO_MIN = 150
        self.SERVO_MAX = 600
        self.SERVO_NEUTRAL = (self.SERVO_MAX + self.SERVO_MIN) // 2
        
        # Motor ports (these can be adjusted during testing)
        self.MOTOR_PORTS = {
            'front_left': 0,
            'front_right': 1,
            'back_left': 2,
            'back_right': 3
        }
        
        # Motor direction flags (True = reversed)
        self.MOTOR_REVERSED = {
            'front_left': False,
            'front_right': False,
            'back_left': False,
            'back_right': False
        }
        
    def set_motor(self, port, speed):
        """
        Set motor speed (-1.0 to 1.0)
        """
        # Reverse speed if motor is configured as reversed
        for motor_name, motor_port in self.MOTOR_PORTS.items():
            if motor_port == port and self.MOTOR_REVERSED[motor_name]:
                speed = -speed
        
        # Convert -1.0 to 1.0 range to servo pulse range
        pulse = int(self.SERVO_NEUTRAL + (speed * (self.SERVO_MAX - self.SERVO_MIN) / 2))
        pulse = max(self.SERVO_MIN, min(self.SERVO_MAX, pulse))
        self.pwm.set_pwm(port, 0, pulse)
    
    def drive(self, x, theta):
        """
        Drive using arcade drive
        x: forward/backward (-1.0 to 1.0)
        theta: rotation (-1.0 to 1.0)
        """
        left = x + theta
        right = x - theta
        
        # Normalize speeds if they exceed [-1, 1]
        max_magnitude = max(abs(left), abs(right))
        if max_magnitude > 1.0:
            left /= max_magnitude
            right /= max_magnitude
            
        # Set motor speeds
        self.set_motor(self.MOTOR_PORTS['front_left'], left)
        self.set_motor(self.MOTOR_PORTS['back_left'], left)
        self.set_motor(self.MOTOR_PORTS['front_right'], right)
        self.set_motor(self.MOTOR_PORTS['back_right'], right)

def cmd_callback(sample):
    try:
        data = json.loads(sample.payload.decode('utf-8'))
        drivebase.drive(data['x'], data['theta'])
    except Exception as e:
        print(f"Error processing command: {e}")

if __name__ == "__main__":
    # Initialize drivebase
    drivebase = DriveBase()
    
    # Initialize Zenoh
    session = zenoh.open()
    sub = session.declare_subscriber('robot/cmd', cmd_callback)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Stop all motors
        drivebase.drive(0, 0)
        session.close()
