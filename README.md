
# README for Vehicle Collision Detection and Avoidance System

## Project Overview

This project aims to develop an AI-based system for detecting and avoiding potential vehicle collisions. The system focuses on detecting moving vehicles, estimating their distance, and predicting collisions by analyzing vehicle motion and depth data. The goal is to enhance the safety of autonomous navigation systems, especially in dynamic traffic environments.

### Current Features
- **Distance Calculation**: The system calculates the distance between the user's vehicle and other vehicles in the vicinity.
- **Collision Detection**: It detects potential collisions by identifying vehicles moving toward the user's vehicle and evaluating their distance using depth estimation and object tracking.
  
### Future Features (Planned)
- Integration of collision avoidance strategies.
- Enhanced detection of stationary and novel objects.
- Improved vehicle trajectory prediction.

---

## How It Works

### Key Components

1. **YOLOv8 for Object Detection**:
   - The system uses YOLOv8 to detect known objects such as vehicles in each video frame.
   - The detected vehicles are then tracked using a Kalman filter.

2. **Monodepth2 for Depth Estimation**:
   - Monodepth2 is used to estimate the distance (depth) of detected objects from the user's vehicle.

3. **Kalman Filter for Object Tracking**:
   - A Kalman filter is applied to track objects across frames and predict their future positions based on their motion.

4. **Optical Flow for Movement Detection**:
   - Optical flow is used to estimate the movement of objects between consecutive frames. This helps to distinguish between moving and stationary objects.

5. **Collision Detection**:
   - The system checks for potential collisions by evaluating the depth (distance) of moving objects and their trajectory. If an object is moving toward the vehicle and is within a critical distance, a collision warning is triggered.

6. **Novel Object Detection (Energy-based Method)**:
   - The system uses a transformer-based model (DINO) to detect novel or unknown objects using an energy-based threshold.

---

## Installation

### Prerequisites

- Python 3.8 or above

### Required Libraries
- OpenCV
- NumPy
- PyTorch
- FilterPy
- SciPy
- Ultralytics YOLOv8
- HuggingFace Transformers

Make sure to download the necessary pretrained models:
- YOLOv8: `yolov8n.pt`
- Monodepth2: `encoder.pth` and `depth.pth`
- DINO: Pretrained models from HuggingFace (`facebook/dino-vitb16`)

---

## Usage

1. Place the video frames in a folder (e.g., `./data/0007`).
2. Run the script:

```bash
python main.py
```

3. The system will process each frame, detect vehicles, estimate their depth, and check for potential collisions.
   
4. Press 'Q' to exit the program.

---

## Next Steps

- Implement real-time collision avoidance mechanisms.
- Expand the detection to recognize various types of vehicles and obstacles in rural and urban settings.

---

## Credits

This project uses the following open-source tools:
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
- [DINO Object Detection](https://huggingface.co/facebook/dino-vitb16)
