import numpy as np
import cv2
import os
import json
from skimage import segmentation
from skimage.measure import find_contours
from itertools import combinations
from ultralytics import YOLO
from tracker import Tracker  # Import your custom Tracker class
import torch

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize the custom tracker
tracker = Tracker()

# Function to calculate minimum distance between boundary pixels of masks
def calculate_min_distance(mask1, mask2):
    contours1 = find_contours(mask1, 0.5)
    contours2 = find_contours(mask2, 0.5)
    if len(contours1) == 0 or len(contours2) == 0:
        return np.inf
    boundary1 = np.vstack(contours1).astype(int)
    boundary2 = np.vstack(contours2).astype(int)
    distances = np.sqrt(np.sum((boundary1[:, np.newaxis, :] - boundary2) ** 2, axis=2))
    if distances.size == 0:
        return np.inf
    return np.min(distances)

# Function to calculate the centroid of a mask
def calculate_centroid(mask):
    indices = np.argwhere(mask)
    centroid = np.mean(indices, axis=0)
    return centroid

# Function to check bounding box overlap percentage
def bounding_box_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    overlap_percentage = intersection_area / float(area1 + area2 - intersection_area)
    return overlap_percentage

# Function to calculate adaptive threshold
def calculate_adaptive_threshold(gap_distances):
    mean_gap_distance = np.mean(gap_distances)
    threshold = mean_gap_distance * 0.1
    return threshold

# Function to calculate speed from trajectory
def calculate_speed(trajectory):
    if len(trajectory) < 2:
        return 0
    distances = [np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])) for i in range(1, len(trajectory))]
    return np.mean(distances)

# Check if CUDA is available and use it if possible
use_cuda = torch.cuda.is_available()
if use_cuda:
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

# Folder containing the video frame folders
video_frames_root = r'C:\Users\akhil\Akhil_work\Baseline\CCTV'

# Initialize variables
trajectories = {}
speeds = {}
near_accident_count = 0
adaptive_threshold = None
accident_cars = set()
detected_accidents = []

# Overall metrics
overall_tp = 0
overall_fp = 0
overall_fn = 0
overall_tn = 0  # Add true negatives

# Process frames from each video folder
video_folders = [os.path.join(video_frames_root, folder) for folder in os.listdir(video_frames_root) if os.path.isdir(os.path.join(video_frames_root, folder))]

for video_folder in video_folders:
    annotations_file = os.path.join(video_folder, 'frames.json')
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    frame_files = sorted([file for file in os.listdir(video_folder) if file.endswith('.jpg')])
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(video_folder, frame_file)
        if not os.path.isfile(frame_path):
            continue

        # Print processing message
        print(f"Processing frame: {frame_file}")

        frame = cv2.imread(frame_path)
        original_frame = frame.copy()

        # Perform object detection
        results = model(frame)

        # Extract bounding boxes and confidences
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy() if use_cuda else result.xyxy[0].numpy()
            conf = result.conf[0].cpu().numpy() if use_cuda else result.conf[0].numpy()
            cls = result.cls[0].cpu().numpy() if use_cuda else result.cls[0].numpy()
            if int(cls) == 2:  # class 2 is for car in COCO dataset
                detections.append([x1, y1, x2, y2, conf])

        # Update the tracker with the current frame and detections
        tracker.update(frame, detections)
        tracked_objects = tracker.tracks

        for track in tracked_objects:
            track_id = track.track_id
            bbox = track.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id not in trajectories:
                trajectories[track_id] = []
                speeds[track_id] = []
            if len(trajectories[track_id]) > 0:
                previous_centroid = trajectories[track_id][-1]
                speed = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))
                speeds[track_id].append(speed)
            trajectories[track_id].append(centroid)

        # Apply SLIC and extract segments
        segments = segmentation.slic(frame, n_segments=100, compactness=10, sigma=1)

        # Create a mask for car segments
        car_masks = {}
        for car_id, (x1, y1, x2, y2, conf) in enumerate(detections):
            mask = np.zeros(segments.shape, dtype=bool)
            mask[int(y1):int(y2), int(x1):int(x2)] = True
            car_masks[car_id] = mask

        # Darken non-car segments
        for segment_id in np.unique(segments):
            if not any(np.any(car_masks[car_id][segments == segment_id]) for car_id in car_masks):
                frame[segments == segment_id] = frame[segments == segment_id] * 0.3  # darken the segment

        # Calculate gap distances and adaptive threshold using the first frame
        if idx == 0:
            gap_distances = []
            for (id1, mask1), (id2, mask2) in combinations(car_masks.items(), 2):
                min_distance = calculate_min_distance(mask1, mask2)
                gap_distances.append(min_distance)
            adaptive_threshold = calculate_adaptive_threshold(gap_distances)
            print(f"Adaptive threshold set to: {adaptive_threshold:.2f}")

        # Check for near-accidents
        near_accident_detected = False
        for (id1, traj1), (id2, traj2) in combinations(trajectories.items(), 2):
            if id1 in car_masks and id2 in car_masks:  # Check if both IDs are in car_masks
                if len(traj1) > 1 and len(traj2) > 1:
                    speed1 = calculate_speed(traj1)
                    speed2 = calculate_speed(traj2)
                    centroid_distance = np.linalg.norm(np.array(traj1[-1]) - np.array(traj2[-1]))
                    min_distance = calculate_min_distance(car_masks[id1], car_masks[id2])
                    overlap_percentage = bounding_box_overlap(detections[id1][:4], detections[id2][:4])

                    # Check for cars already in accident_cars list
                    if id1 in accident_cars and id2 in accident_cars:
                        if overlap_percentage > 0.2:
                            near_accident_detected = True
                    else:
                        if min_distance < adaptive_threshold and overlap_percentage > 0.2 and speed1 > 0 and speed2 > 0:
                            near_accident_detected = True

                    if near_accident_detected:
                        detected_accidents.append(frame_file)
                        near_accident_count += 1
                        print(f"Near-accident detected between Car {id1} and Car {id2}!")
                        accident_cars.add(id1)
                        accident_cars.add(id2)
                        break  # Exit the loop if an accident is detected

        # Draw bounding boxes and annotations on the frame
        for track in tracked_objects:
            track_id = track.track_id
            bbox = track.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            speed = np.mean(speeds[track_id]) if len(speeds[track_id]) > 0 else 0
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow bounding box
            cv2.putText(original_frame, f'id:{track_id} Speed: {speed:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save the frame if near-accident is detected
        if near_accident_detected:
            near_accident_frame_path = os.path.join(video_folder, 'near_accidents', frame_file)
            os.makedirs(os.path.dirname(near_accident_frame_path), exist_ok=True)
            cv2.imwrite(near_accident_frame_path, original_frame)

        # Save the processed frame
        processed_frame_path = os.path.join(video_folder, 'processed_frames', frame_file)
        os.makedirs(os.path.dirname(processed_frame_path), exist_ok=True)
        cv2.imwrite(processed_frame_path, original_frame)

        # Print processing complete message
        print(f"Frame {idx + 1} processed - Found {near_accident_count} number of accidents")

    # Calculate accuracy for the current video
    tp = 0
    fp = 0
    fn = 0

    for annotation in annotations:
        frame = annotation["frame"]
        accident = annotation["accident"]
        if accident and frame in detected_accidents:
            tp += 1
        elif not accident and frame in detected_accidents:
            fp += 1
        elif accident and frame not in detected_accidents:
            fn += 1

    tn = len(annotations) - (tp + fp + fn)  # Calculate true negatives

    overall_tp += tp
    overall_fp += fp
    overall_fn += fn
    overall_tn += tn

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"Video folder: {video_folder}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

# Calculate overall metrics
overall_precision = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp > 0 else 0
overall_recall = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn > 0 else 0
overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
overall_accuracy = (overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn) if (overall_tp + overall_tn + overall_fp + overall_fn) > 0 else 0

print(f"Overall Precision: {overall_precision:.2f}")
print(f"Overall Recall: {overall_recall:.2f}")
print(f"Overall F1 Score: {overall_f1_score:.2f}")
print(f"Overall Accuracy: {overall_accuracy:.2f}")
