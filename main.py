import cv2
import numpy as np
import torch
import os
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from models.monodepth2 import networks
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

yolo_model = YOLO('exps/yolo/yolov8n.pt')

encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

encoder_checkpoint = torch.load("exps/monodepth2/encoder.pth", map_location='cpu')
depth_checkpoint = torch.load("exps/monodepth2/depth.pth", map_location='cpu')

filtered_encoder_checkpoint = {k: v for k, v in encoder_checkpoint.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_encoder_checkpoint)
filtered_depth_checkpoint = {k: v for k, v in depth_checkpoint.items() if k in depth_decoder.state_dict()}
depth_decoder.load_state_dict(filtered_depth_checkpoint)

encoder.eval()
depth_decoder.eval()

processor = DetrImageProcessor.from_pretrained("facebook/dino-vitb16")
dino_model = DetrForObjectDetection.from_pretrained("facebook/dino-vitb16")
dino_model.eval()

class ObjectTracker:
    """Class to track objects using Kalman Filter."""
    
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.moving_history = {}

    def update(self, detections):
        """Update Kalman filters with detected objects."""
        
        for det in detections:
            x = (det[0] + det[2]) / 2
            y = (det[1] + det[3]) / 2
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x = np.array([x, y, 0, 0])
            kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.trackers[self.next_id] = kf
            self.moving_history[self.next_id] = False  
            self.next_id += 1
        return self.trackers

    def update_moving_status(self, obj_id, velocity):
        """Update the moving status of an object."""

        self.moving_history[obj_id] = velocity > 5

    def assign_detections_to_trackers(self, detections, trackers):
        """Assign detected objects to trackers based on cost matrix."""

        if not trackers:
            return np.empty((0, 2), dtype=int)

        cost_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers.values()):
                x1, y1 = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
                x2, y2 = trk.x[0], trk.x[1]
                cost_matrix[d, t] = np.linalg.norm([x1 - x2, y1 - y2])

        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_inds, col_inds)))

def detect_known_objects(frame):
    """Detect known objects using YOLOv8."""

    results = yolo_model(frame)
    return results

def detect_novel_objects(frame, processor, model, threshold=-10.0):
    """Detect novel objects using Energy-based methods."""

    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0]
    boxes = outputs.pred_boxes[0]
    novel_boxes = []
    energy = torch.logsumexp(logits, dim=-1)

    for i, e in enumerate(energy):
        if e < threshold:
            novel_boxes.append(boxes[i].detach().cpu().numpy())
    
    return novel_boxes

def estimate_depth(frame):
    """Estimate depth map using Monodepth2."""

    frame_resized = cv2.resize(frame, (640, 192))
    frame_tensor = torch.from_numpy(frame_resized).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        features = encoder(frame_tensor)
        outputs = depth_decoder(features)
    depth_map = outputs[("disp", 0)][0, 0].cpu().numpy()
    cv2.imshow('Depth Map', depth_map)
    return depth_map

def estimate_optical_flow(prev_frame, curr_frame):
    """Estimate optical flow between two frames."""

    if prev_frame is None or curr_frame is None:
        return None
    return cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def check_for_moving_objects(flow, detections, threshold=13):
    """Check for moving objects using optical flow and flow magnitude."""

    flow_magnitude = np.linalg.norm(flow, axis=2)
    background_motion = np.median(flow_magnitude)
    moving_objects = []

    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det)
        flow_patch = flow_magnitude[y_min:y_max, x_min:x_max]
        avg_magnitude = np.mean(flow_patch)
        if avg_magnitude - background_motion > threshold:
            moving_objects.append(det)
    return moving_objects

def check_for_collision(moving_objects, depth_map, threshold=1.0):
    """Check for potential collisions by comparing object positions with the depth map."""

    for obj_id, obj_box in enumerate(moving_objects):
        x_min, y_min, x_max, y_max = map(int, obj_box)
        depths = [depth_map[y, x] for x in range(x_min, x_max) for y in range(y_min, y_max) if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]]
        if depths and np.min(depths) < threshold:
            print(f"Collision Warning: Moving object {obj_id} is close! Closest distance: {np.min(depths):.2f} meters")
            return True
    return False

def avoid_collision(frame, moving_objects):
    """Detect potential collisions and trigger collision avoidance action."""

    depth_map = estimate_depth(frame)
    if check_for_collision(moving_objects, depth_map, threshold=0.5):
        cv2.putText(frame, "Collision Warning!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Collision avoidance action triggered!")
        return True
    return False

def process_frames_with_avoidance_and_flow(folder_path):
    """Process frames, detect moving objects, check for collisions, and trigger avoidance if necessary."""
    
    object_tracker = ObjectTracker()
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    prev_frame_gray = None
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            continue
        
        known_objects = detect_known_objects(frame)
        detections = known_objects[0].boxes.xyxy.cpu().numpy()

        tracked_objects = object_tracker.update(detections)
        for obj_id, kf in tracked_objects.items():
            vx, vy = kf.x[2], kf.x[3]
            speed = np.sqrt(vx**2 + vy**2)
            object_tracker.update_moving_status(obj_id, speed)

        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            flow = estimate_optical_flow(prev_frame_gray, curr_frame_gray)
            moving_detections = check_for_moving_objects(flow, detections)
            for obj_box in moving_detections:
                x_min, y_min, x_max, y_max = map(int, obj_box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if avoid_collision(frame, moving_detections):
                print("Taking action to avoid collision")

            novel_objects = detect_novel_objects(frame, processor, dino_model, threshold=-25.0)
            if novel_objects:
                print(f"Novel objects detected: {novel_objects}")
            for box in novel_objects:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        prev_frame_gray = curr_frame_gray
        cv2.imshow('Frame Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    folder_path = './data/0007'
    process_frames_with_avoidance_and_flow(folder_path)
