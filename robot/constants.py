# Motor ports (these can be adjusted during testing)
MOTOR_PORTS = {
    'front_left': 3,
    'front_right': 2,
    'back_left': 1,
    'back_right': 0
}

# Motor direction flags (True = reversed)
MOTOR_REVERSED = {
    'front_left': False,
    'front_right': True,
    'back_left': False,
    'back_right': True,
}

# Servo/Motor PWM Configuration
SERVO_MIDDLE = 365 
SERVO_MIN = SERVO_MIDDLE - 225
SERVO_MAX = SERVO_MIDDLE + 225
SERVO_NEUTRAL = (SERVO_MAX + SERVO_MIN) // 2

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Zenoh keys
CAMERA_FRAME_KEY = "robot/camera/frame"
ROBOT_TWIST_CMD_KEY = "robot/cmd"
FLIP_FRAME = False