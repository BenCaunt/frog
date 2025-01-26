## Frog 

A simple mobile robot designed for developing autonomous machine learning agents.

### Overview

Frog is a tank-drive mobile robot platform with a webcam, designed for collecting exploration datasets and developing autonomous agents. The system uses Zenoh for communication between components and includes tools for teleoperating the robot, logging datasets, and replaying collected data.

### System Architecture

The system consists of three main components:
- Robot: Hardware control and sensor publishing
- Client: Teleoperation and dataset logging
- Utils: Dataset visualization and playback

### Zenoh Topics

The system uses the following Zenoh topics for communication:

| Topic | Description | Data Format |
|-------|-------------|-------------|
| `robot/cmd` | Drive commands | JSON: `{"x": float, "theta": float}` |
| `robot/camera/frame` | Camera frames | JPEG encoded bytes |

- `x`: Forward/backward velocity (-1.0 to 1.0)
- `theta`: Rotational velocity (-1.0 to 1.0)

### Usage

#### Robot Setup

1. Connect to the robot via SSH
2. Launch the robot software:
```bash
python robot/launch.py
```

This will start:
- Motor control (`drivebase.py`)
- Camera publishing (`webcam_publisher.py`)

#### Teleoperation

To teleoperate the robot using a gamepad:

```bash
python client/teleop.py
```

Controls:
- Left stick Y-axis: Forward/backward
- Right stick X-axis: Rotation
- Deadband: 0.1
- Max linear speed: 1.0
- Max angular speed: 0.8

#### Dataset Collection

To record a dataset while teleoperating:

```bash
python client/dataset_logger.py
```

This will:
- Create a timestamped directory in `data/`
- Save camera frames as JPEGs
- Record twist commands
- Generate metadata and sequence files

Dataset structure:
```
data/YYYYMMDD_HHMMSS/
├── images/
│   └── frame_*.jpg
├── metadata.json
└── sequences.json
```

#### Dataset Playback

To visualize a recorded dataset:

```bash
python utils/dataset_player.py path/to/dataset --speed 1.0
```

This will:
- Display the camera feed
- Plot twist commands over time
- Allow adjusting playback speed

### Dependencies

- Python 3.8+
- OpenCV
- Pygame (for teleop)
- Zenoh
- Rerun (for visualization)
- Adafruit PCA9685 (for motor control)

### Development

The codebase is organized as follows:
```
frog/
├── robot/          # Robot-side code
├── client/         # Operator-side code
└── utils/          # Development tools
```

