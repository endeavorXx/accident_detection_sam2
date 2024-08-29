import os
import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

final_ll = []
detection_threshold = 0.4

trajectories = {}

near_accidents = {}
potential_accidents = {}

def calculate_speed_and_direction(traj):
    if len(traj) < 2:
        return (0, 0)
    dx = traj[-1][0] - traj[-2][0]
    dy = traj[-1][1] - traj[-2][1]
    speed = np.sqrt(dx ** 2 + dy ** 2)
    direction = np.arctan2(dy, dx)
    return speed, direction

def detect_near_accident(traj1, traj2, bbox1, bbox2, threshold=30, iou_threshold=0.1):
    distance_condition = np.linalg.norm(np.array(traj1[-1]) - np.array(traj2[-1])) < 50
    final_ll.append(np.linalg.norm(np.array(traj1[-1]) - np.array(traj2[-1])))
    iou_condition = calculate_iou(bbox1, bbox2) > iou_threshold
    return distance_condition or iou_condition

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area

    iou = inter_area / union_area
    return iou

def is_within_frame(center, frame_shape):
    x, y = center
    h, w = frame_shape[:2]
    return 0 <= x < w and 0 <= y < h

def calculate_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def process_video(video_path, model, tracker, detection_threshold=0.4):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        return 0  # If the video cannot be read, classify as no accident

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    trajectories = {}
    near_accidents = {}
    potential_accidents = {}

    accident_detected = False

    while ret:
        results = model(frame)
        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if class_id != 0:
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

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if is_within_frame(center, frame.shape):
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append(center)
            else:
                if track_id in trajectories:
                    del trajectories[track_id]

        visible_track_ids = {track.track_id for track in tracker.tracks}
        trajectories = {track_id: traj for track_id, traj in trajectories.items() if track_id in visible_track_ids}
        near_accidents = {key: expire_time for key, expire_time in near_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}
        potential_accidents = {key: expire_time for key, expire_time in potential_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}

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
                        accident_detected = True

        if accident_detected:
            break

        frame_count += 1
        ret, frame = cap.read()

    cap.release()
    return 1 if accident_detected else 0

def evaluate_videos(accident_folder, non_accident_folder, model, tracker):
    ground_truth_labels = []
    predicted_labels = []

    count=0

    for video in os.listdir(accident_folder):
        print("Video processing for accident - ",count)
        count+=1
        video_path = os.path.join(accident_folder, video)
        ground_truth_labels.append(1)
        predicted_labels.append(process_video(video_path, model, tracker))
    
    count=0
    for video in os.listdir(non_accident_folder):
        print("Video processing for non_accident - ",count)
        count+=1
        video_path = os.path.join(non_accident_folder, video)
        ground_truth_labels.append(0)
        predicted_labels.append(process_video(video_path, model, tracker))
    
    print("Ground truth Labels : ")
    print(ground_truth_labels)

    print("Predicted Labels : ")
    print(predicted_labels)

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels)
    recall = recall_score(ground_truth_labels, predicted_labels)
    f1 = f1_score(ground_truth_labels, predicted_labels)
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

# Paths to your video folders
accident_folder = r"C:\Users\akhil\Akhil_work\deep_sort\__MACOSX\test\Backend"
non_accident_folder = r"C:\Users\akhil\Akhil_work\deep_sort\__MACOSX\test\non_accident_120"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# Evaluate videos
evaluate_videos(accident_folder, non_accident_folder, model, tracker)
