import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

# Paths
video_path = r"C:\\Users\\akhil\\Downloads\\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4"
video_out_path = os.path.join('.', 'out4.mp4')

# Video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# Colors for bounding boxes and trajectories
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

# Detection threshold
detection_threshold = 0.5

# Dictionary to hold trajectories
trajectories = {}

# Dictionary to hold near-accident messages and their timestamps
near_accidents = {}

# Frame count and time variables
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 5)  # 5 seconds interval

def calculate_speed_and_direction(traj):
    if len(traj) < 2:
        return (0, 0)
    dx = traj[-1][0] - traj[-2][0]
    dy = traj[-1][1] - traj[-2][1]
    speed = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)
    return speed, direction

def detect_near_accident(traj1, traj2, bbox1, bbox2, threshold=30, iou_threshold=0.3):
    for i in range(min(len(traj1), len(traj2))):
        if np.linalg.norm(np.array(traj1[i]) - np.array(traj2[i])) < threshold and calculate_iou(bbox1, bbox2) > iou_threshold:
            return True
    return False

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union area
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area

    iou = inter_area / union_area
    return iou

def is_within_frame(center, frame_shape):
    x, y = center
    h, w = frame_shape[:2]
    return 0 <= x < w and 0 <= y < h

while ret:
    results = model(frame)
    detections = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if class_id == 2:  # Class ID for cars
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

    tracker.update(frame, detections)

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id

        # Update trajectories
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if is_within_frame(center, frame.shape):
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append(center)

            # Draw bounding box with track ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            if track_id in trajectories:
                del trajectories[track_id]

    # Remove trajectories and near-accident messages for vehicles no longer in frame
    visible_track_ids = {track.track_id for track in tracker.tracks}
    trajectories = {track_id: traj for track_id, traj in trajectories.items() if track_id in visible_track_ids}
    near_accidents = {key: expire_time for key, expire_time in near_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}

    # Draw trajectories
    for track_id, points in trajectories.items():
        if len(points) > 1:
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(frame, pt1, pt2, colors[track_id % len(colors)], 2)

    # Detect and label near-accidents
    current_time = frame_count / fps
    for t_id1, traj1 in trajectories.items():
        for t_id2, traj2 in trajectories.items():
            if t_id1 >= t_id2:
                continue
            track1 = next((track for track in tracker.tracks if track.track_id == t_id1), None)
            track2 = next((track for track in tracker.tracks if track.track_id == t_id2), None)
            if track1 and track2:
                bbox1 = [int(x) for x in track1.bbox]
                bbox2 = [int(x) for x in track2.bbox]
                speed1, direction1 = calculate_speed_and_direction(traj1)
                speed2, direction2 = calculate_speed_and_direction(traj2)
                if detect_near_accident(traj1, traj2, bbox1, bbox2) and abs(direction1 - direction2) < np.pi / 4:
                    # Store near accident message with expiration time
                    near_accidents[(t_id1, t_id2)] = current_time + 5

    # Draw near accident messages and remove expired ones
    expired_keys = []
    for (t_id1, t_id2), expire_time in near_accidents.items():
        if current_time > expire_time:
            expired_keys.append((t_id1, t_id2))
        else:
            # Get the last known positions of the tracks
            if t_id1 in trajectories and t_id2 in trajectories:
                pt1 = trajectories[t_id1][-1]
                pt2 = trajectories[t_id2][-1]
                midpoint = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
                cv2.putText(frame, "Near Accident!", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Remove expired near-accident messages
    for key in expired_keys:
        del near_accidents[key]

    cap_out.write(frame)

    # Save an image every 5 seconds
    if frame_count % interval_frames == 0:
        cv2.imwrite(f'frame_{frame_count}.png', frame)

    frame_count += 1
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
