# Motor ports (these can be adjusted during testing)
MOTOR_PORTS = {
    'front_left': 0,
    'front_right': 1,
    'back_left': 3,
    'back_right': 2
}

# Motor direction flags (True = reversed)
MOTOR_REVERSED = {
    'front_left': False,
    'front_right': True,
    'back_left': False,
    'back_right': True,
}

# Servo/Motor PWM Configuration
SERVO_MIDDLE = 400 
SERVO_MIN = SERVO_MIDDLE - 225
SERVO_MAX = SERVO_MIDDLE + 225
SERVO_NEUTRAL = (SERVO_MAX + SERVO_MIN) // 2

# Camera configuration
CAMERA_WIDTH = 3480
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# Zenoh keys
CAMERA_FRAME_KEY = "robot/camera/frame"
