# Motor ports (these can be adjusted during testing)
MOTOR_PORTS = {
    'front_left': 0,
    'front_right': 1,
    'back_left': 2,
    'back_right': 3
}

# Motor direction flags (True = reversed)
MOTOR_REVERSED = {
    'front_left': False,
    'front_right': False,
    'back_left': False,
    'back_right': False
}

# Servo/Motor PWM Configuration
SERVO_MIDDLE = 400 
SERVO_MIN = SERVO_MIDDLE - 225
SERVO_MAX = SERVO_MIDDLE + 225
SERVO_NEUTRAL = (SERVO_MAX + SERVO_MIN) // 2
