import os
import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

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
        return 0, 0, 0, 0  # If the video cannot be read, return zeros

    frame_count = 0

    detection_latency = 0
    tracking_latency = 0
    total_latency = 0

    while ret:
        start_time = time.time()
        
        detection_start_time = time.time()
        results = model(frame)
        detection_end_time = time.time()
        detection_latency += (detection_end_time - detection_start_time)
        
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

        tracking_start_time = time.time()
        tracker.update(frame, detections)
        tracking_end_time = time.time()
        tracking_latency += (tracking_end_time - tracking_start_time)

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

        frame_count += 1
        ret, frame = cap.read()

        end_time = time.time()
        total_latency += (end_time - start_time)

    cap.release()
    return detection_latency, tracking_latency, total_latency, frame_count

def measure_latencies(video_folder, model, tracker, num_videos=20):
    video_files = os.listdir(video_folder)
    if len(video_files) < num_videos:
        num_videos = len(video_files)
    sampled_videos = random.sample(video_files, num_videos)

    total_detection_latency = 0
    total_tracking_latency = 0
    total_processing_latency = 0
    total_frames = 0

    for video in sampled_videos:
        video_path = os.path.join(video_folder, video)
        detection_latency, tracking_latency, processing_latency, frame_count = process_video(video_path, model, tracker)
        
        total_detection_latency += detection_latency
        total_tracking_latency += tracking_latency
        total_processing_latency += processing_latency
        total_frames += frame_count

    avg_detection_latency = total_detection_latency / total_frames
    avg_tracking_latency = total_tracking_latency / total_frames
    avg_total_latency = total_processing_latency / total_frames

    print("Average Detection Latency per Frame: {:.4f} seconds".format(avg_detection_latency))
    print("Average Tracking Latency per Frame: {:.4f} seconds".format(avg_tracking_latency))
    print("Average Total Latency per Frame: {:.4f} seconds".format(avg_total_latency))

# Paths to your video folder
video_folder = r"C:\Users\akhil\Akhil_work\deep_sort\__MACOSX\test\Mixed_accident_non_accident"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# Measure latencies for a random sample of 20 videos
measure_latencies(video_folder, model, tracker, num_videos=20)
