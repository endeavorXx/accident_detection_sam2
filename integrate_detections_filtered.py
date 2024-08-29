r'''import os
import cv2
import random
import time
import numpy as np
from tracker import Tracker
from ultralytics import YOLO
import torch
import json

from yolofunctions import calculate_distance, calculate_iou, calculate_speed_and_direction, detect_near_accident, is_within_frame

video_base_path = r"C:\Users\akhil\OneDrive\Desktop\Test_perspective_divided\Test\Drone"
accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Accident_output_frames"
non_accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Non_Accident_output_frames"

#json_files_path = r"C:\Users\akhil\Akhil_work\Baseline\Output\Output_accident_preprocessed\preprocess_binary_10x"
json_files_path = r"C:\Users\akhil\Akhil_work\Baseline\Output\Output_accident_preprocessed_CCTV\preprocess_binary_10x"


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8m.pt")
detection_threshold = 0.4

total_detection_latency = 0
total_tracking_latency = 0
total_latency = 0
total_frames = 0
total_detections = 0
total_tracks = 0
processed_frames = 0
skipped_frames = 0

def process_frames_from_folder(frames_folder, json_file_path):
    global total_detection_latency, total_tracking_latency, total_latency, total_frames, total_detections, total_tracks, processed_frames, skipped_frames

    # Initialize tracker
    tracker = Tracker()

    # Dictionary to hold trajectories
    trajectories = {}

    # Dictionary to hold near-accident messages and their timestamps
    near_accidents = {}

    # Load JSON file for the video
    if not os.path.exists(json_file_path):
        print(f"JSON file not found: {json_file_path}")
        return False

    with open(json_file_path, 'r') as f:
        frame_flags = json.load(f)

    # Frame count
    frame_count = 0

    accident_detected = False

    frames = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    frames.sort()  # Ensure frames are processed in order

    # Filter frame files based on JSON flags
    filtered_frame_files = [frame for frame in frames if frame_flags.get(os.path.basename(frame), False)]
    skipped_frames += len(frames) - len(filtered_frame_files)  # Count skipped frames
    if not filtered_frame_files:
        print(f"No frames to process in folder: {frames_folder}")
        return False

    for frame_path in filtered_frame_files:
        processed_frames += 1
        frame = cv2.imread(frame_path)
        start_time = time.time()

        # Object detection
        detection_start_time = time.time()
        results = model(frame)
        detection_end_time = time.time()
        detection_latency = detection_end_time - detection_start_time
        total_detection_latency += detection_latency

        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if class_id != 0:  # Class ID for cars
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    if score > detection_threshold:
                        detections.append([x1, y1, x2, y2, score])
                        total_detections += 1

        # Tracking
        tracking_start_time = time.time()
        tracker.update(frame, detections)
        tracking_end_time = time.time()
        tracking_latency = tracking_end_time - tracking_start_time
        total_tracking_latency += tracking_latency

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

        total_tracks += len(tracker.tracks)

        # Remove trajectories and near-accident messages for vehicles no longer in frame
        visible_track_ids = {track.track_id for track in tracker.tracks}
        trajectories = {track_id: traj for track_id, traj in trajectories.items() if track_id in visible_track_ids}
        near_accidents = {key: expire_time for key, expire_time in near_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}

        # Detect and label near-accidents
        current_time = frame_count / 10  # Assuming 10 FPS
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
                        accident_detected = True

        # Remove expired near-accident messages
        expired_keys = []
        for (t_id1, t_id2), expire_time in near_accidents.items():
            if current_time > expire_time:
                expired_keys.append((t_id1, t_id2))

        for key in expired_keys:
            del near_accidents[key]

        frame_count += 1
        total_frames += 1
        end_time = time.time()
        total_latency += (end_time - start_time)

    return accident_detected

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

def process_videos(video_folder, json_folder, is_accident):
    global true_positive, false_positive, true_negative, false_negative

    for video in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video)
        json_file_path = os.path.join(json_folder, f"{video}.json")
        print(f"Processing video frames in folder: {video_path}")
        accident_detected = process_frames_from_folder(video_path, json_file_path)

        if is_accident:
            if accident_detected:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if accident_detected:
                false_positive += 1
            else:
                true_negative += 1

# Process accident video frames
process_videos(accident_frames_folder, json_files_path, True)

# Process non-accident video frames
process_videos(non_accident_frames_folder, json_files_path, False)

# Calculate accuracies
if total_frames > 0:
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate throughput
    detection_throughput = total_frames / total_detection_latency if total_detection_latency > 0 else 0
    tracking_throughput = total_frames / total_tracking_latency if total_tracking_latency > 0 else 0
    total_throughput = total_frames / total_latency if total_latency > 0 else 0

    print(f"Mode: FRAME PROCESSING")
    print(f"Device used: {device}")
    print(f"Total frames: {total_frames}")
    print(f"Processed frames: {processed_frames}")
    print(f"Skipped frames: {skipped_frames}")
    print("Average Detection Latency per Frame: {:.4f} ms".format(total_detection_latency / total_frames * 1000))
    print("Average Tracking Latency per Frame: {:.4f} ms".format(total_tracking_latency / total_frames * 1000))
    print("Average Total Latency per Frame: {:.4f} ms".format(total_latency / total_frames * 1000))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1_score * 100:.2f}%")
    print("Detection Throughput: {:.4f} frames/s".format(detection_throughput))
    print("Tracking Throughput: {:.4f} frames/s".format(tracking_throughput))
    print("Total Throughput: {:.4f} frames/s".format(total_throughput))
else:
    print("No frames processed.")
'''

import os
import cv2
import time
import numpy as np
from tracker import Tracker
from ultralytics import YOLO
import torch
import json
from tqdm import tqdm
import logging

from yolofunctions import calculate_distance, calculate_iou, calculate_speed_and_direction, detect_near_accident, is_within_frame

video_base_path = r"C:\Users\akhil\OneDrive\Desktop\Test_perspective_divided\Test\Drone"
accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Accident_output_frames"
non_accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Non_Accident_output_frames"
json_files_path = r"C:\Users\akhil\Akhil_work\Baseline\Output\Output_accident_preprocessed_CCTV\preprocess_binary_10x"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8m.pt")
detection_threshold = 0.4

# Set logging level to WARNING to suppress info messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)

total_detection_latency = 0
total_tracking_latency = 0
total_latency = 0
total_frames = 0
total_detections = 0
total_tracks = 0
processed_frames = 0
skipped_frames = 0
total_video_latency = 0
video_count = 0

def process_frames_from_folder(frames_folder, json_file_path):
    global total_detection_latency, total_tracking_latency, total_latency, total_frames, total_detections, total_tracks, processed_frames, skipped_frames, total_video_latency

    # Initialize tracker
    tracker = Tracker()

    # Dictionary to hold trajectories
    trajectories = {}

    # Dictionary to hold near-accident messages and their timestamps
    near_accidents = {}

    # Load JSON file for the video
    if not os.path.exists(json_file_path):
        print(f"JSON file not found: {json_file_path}")
        return False

    with open(json_file_path, 'r') as f:
        frame_flags = json.load(f)

    frames = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    frames.sort()  # Ensure frames are processed in order

    # Filter frame files based on JSON flags
    filtered_frame_files = [frame for frame in frames if frame_flags.get(os.path.basename(frame), False)]
    skipped_frame_files = [frame for frame in frames if not frame_flags.get(os.path.basename(frame), False)]
    skipped_frames += len(skipped_frame_files)  # Count skipped frames

    # Print skipped frames
    for skipped_frame in skipped_frame_files:
        print(f"Skipped frame: {os.path.basename(skipped_frame)}")

    if not filtered_frame_files:
        print(f"No frames to process in folder: {frames_folder}")
        return False

    frame_detection_latency = 0
    frame_tracking_latency = 0
    frame_total_latency = 0

    video_start_time = time.time()

    for frame_path in filtered_frame_files:
        processed_frames += 1
        frame = cv2.imread(frame_path)
        start_time = time.time()

        # Object detection
        detection_start_time = time.time()
        results = model(frame)
        detection_end_time = time.time()
        detection_latency = detection_end_time - detection_start_time
        total_detection_latency += detection_latency
        frame_detection_latency += detection_latency

        detections = []

        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if class_id != 0:  # Class ID for cars
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    if score > detection_threshold:
                        detections.append([x1, y1, x2, y2, score])
                        total_detections += 1

        # Tracking
        tracking_start_time = time.time()
        tracker.update(frame, detections)
        tracking_end_time = time.time()
        tracking_latency = tracking_end_time - tracking_start_time
        total_tracking_latency += tracking_latency
        frame_tracking_latency += tracking_latency

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

        total_tracks += len(tracker.tracks)

        # Remove trajectories and near-accident messages for vehicles no longer in frame
        visible_track_ids = {track.track_id for track in tracker.tracks}
        trajectories = {track_id: traj for track_id, traj in trajectories.items() if track_id in visible_track_ids}
        near_accidents = {key: expire_time for key, expire_time in near_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}

        # Detect and label near-accidents
        current_time = processed_frames / 10  # Assuming 10 FPS
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

        # Remove expired near-accident messages
        expired_keys = []
        for (t_id1, t_id2), expire_time in near_accidents.items():
            if current_time > expire_time:
                expired_keys.append((t_id1, t_id2))

        for key in expired_keys:
            del near_accidents[key]

        frame_total_latency += (time.time() - start_time)

    video_end_time = time.time()
    video_latency = video_end_time - video_start_time
    total_video_latency += video_latency

    video_frames = len(filtered_frame_files)
    if video_frames > 0:
        video_detection_latency = frame_detection_latency / video_frames * 1000
        video_tracking_latency = frame_tracking_latency / video_frames * 1000
        video_total_latency = frame_total_latency / video_frames * 1000
        video_detection_throughput = video_frames / frame_detection_latency if frame_detection_latency > 0 else 0
        video_tracking_throughput = video_frames / frame_tracking_latency if frame_tracking_latency > 0 else 0
        video_total_throughput = video_frames / frame_total_latency if frame_total_latency > 0 else 0

        print(f"Video: {frames_folder}")
        print(f"Frames: {video_frames}")
        print("Video Detection Latency per Frame: {:.4f} ms".format(video_detection_latency))
        print("Video Tracking Latency per Frame: {:.4f} ms".format(video_tracking_latency))
        print("Video Total Latency per Frame: {:.4f} ms".format(video_total_latency))
        print("Video Detection Throughput: {:.4f} frames/s".format(video_detection_throughput))
        print("Video Tracking Throughput: {:.4f} frames/s".format(video_tracking_throughput))
        print("Video Total Throughput: {:.4f} frames/s".format(video_total_throughput))

    return True

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

def process_videos(video_folder, json_folder, is_accident):
    global true_positive, false_positive, true_negative, false_negative, video_count, total_video_latency

    video_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
    video_folders = video_folders[:40]  # Limit to 40 videos

    for video in tqdm(video_folders, desc="Processing videos"):
        video_path = os.path.join(video_folder, video)
        json_file_path = os.path.join(json_folder, f"{video}.json")
        print(f"Processing video frames in folder: {video_path}")
        accident_detected = process_frames_from_folder(video_path, json_file_path)

        video_count += 1

        if is_accident:
            if accident_detected:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if accident_detected:
                false_positive += 1
            else:
                true_negative += 1

# Process accident video frames
process_videos(accident_frames_folder, json_files_path, True)

# Process non-accident video frames
process_videos(non_accident_frames_folder, json_files_path, False)

# Calculate accuracies
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate overall throughput
detection_throughput = total_frames / total_detection_latency if total_detection_latency > 0 else 0
tracking_throughput = total_frames / total_tracking_latency if total_tracking_latency > 0 else 0
total_throughput = total_frames / total_latency if total_latency > 0 else 0

# Calculate average latency and throughput per video
average_video_latency = total_video_latency / video_count if video_count > 0 else 0
average_video_throughput = video_count / total_video_latency if total_video_latency > 0 else 0

print(f"Mode: FRAME PROCESSING")
print(f"Device used: {device}")
print("Total frames processed: {}".format(total_frames))
print("Total frames skipped: {}".format(skipped_frames))
if total_frames > 0:
    print("Average Detection Latency per Frame: {:.4f} ms".format(total_detection_latency / total_frames * 1000))
    print("Average Tracking Latency per Frame: {:.4f} ms".format(total_tracking_latency / total_frames * 1000))
    print("Average Total Latency per Frame: {:.4f} ms".format(total_latency / total_frames * 1000))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")
if total_detection_latency > 0:
    print("Detection Throughput: {:.4f} frames/s".format(detection_throughput))
if total_tracking_latency > 0:
    print("Tracking Throughput: {:.4f} frames/s".format(tracking_throughput))
if total_latency > 0:
    print("Total Throughput: {:.4f} frames/s".format(total_throughput))
if video_count > 0:
    print("Average Latency per Video: {:.4f} s".format(average_video_latency))
if total_video_latency > 0:
    print("Average Throughput per Video: {:.4f} videos/s".format(average_video_throughput))
