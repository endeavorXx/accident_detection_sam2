import os
import cv2
import numpy as np
import glob
import json
import time
import torch
from ultralytics import YOLO
from tqdm import tqdm


# Load the YOLOv8 model and specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x.pt').to(device)




def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def check_proximity(boxes, threshold=0.1):
    close_pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if bb_intersection_over_union(boxes[i], boxes[j]) > threshold:
                close_pairs.append((i, j))
    return len(close_pairs) > 0

# Path to the directory containing the frames
frames_dir = r"C:\Users\akhil\OneDrive\Desktop\Test_perspective_divided\Test\CCTV"

accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Accident_output_frames"
non_accident_frames_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Non_Accident_output_frames"

output_dir = r"C:\Users\akhil\Akhil_work\Baseline\Output\Output_accident_preprocessed_CCTV"
preprocess_binary_dir = os.path.join(output_dir, "preprocess_binary_10x")
os.makedirs(preprocess_binary_dir, exist_ok=True)

# List all video folders
video_folders = os.listdir(accident_frames_folder) + os.listdir(non_accident_frames_folder)
start_time = time.time()
total_frames = 0
tumbling_window_size = 10

# Process each video folder
for video_folder in tqdm(video_folders, desc='Processing video folders'):
    if video_folder in os.listdir(accident_frames_folder):
        frames_folder = os.path.join(accident_frames_folder, video_folder)
    else:
        frames_folder = os.path.join(non_accident_frames_folder, video_folder)

    image_files = sorted(glob.glob(os.path.join(frames_folder, '*.jpg')))
    total_frames += len(image_files)
    video_results = {}

    i = 0
    while i < len(image_files):
        window_end = min(i + tumbling_window_size, len(image_files))
        critical_event_detected = False

        for j in range(i, min(window_end, len(image_files))):
            frame = cv2.imread(image_files[j])
            results = model(frame)

            boxes = [(int(obj.xyxy[0].tolist()[0]), int(obj.xyxy[0].tolist()[1]), int(obj.xyxy[0].tolist()[2]), int(obj.xyxy[0].tolist()[3]))
                     for obj in results[0].boxes if obj.cls in [2, 3, 5, 7, 1]]  # Filtering vehicle classes

            if check_proximity(boxes):
                critical_event_detected = True
                break  # Exit early from tumbling window on detection

        if critical_event_detected:
            # Extend the context-aware window up to 20 frames forward from the current index
            context_end = min(j + 2 * tumbling_window_size, len(image_files))
            for k in range(j, context_end):
                video_results[os.path.basename(image_files[k])] = True
            i = context_end  # Move index past the context-aware window
        else:
            # No critical event, move to next tumbling window
            for k in range(i, window_end):
                video_results[os.path.basename(image_files[k])] = False
            i = window_end  # Increment window start

    # Save results
    json_path = os.path.join(preprocess_binary_dir, f'{os.path.basename(video_folder)}.json')
    with open(json_path, 'w') as json_file:
        json.dump(video_results, json_file)

end_time = time.time()
total_latency = end_time - start_time
average_latency_per_frame = total_latency / total_frames  # seconds per frame
average_latency_per_frame_ms = average_latency_per_frame * 1000  # convert to milliseconds

# Calculate throughput
throughput = total_frames / total_latency  # frames per second

print(f'Device: {device}')
print(f'Total latency: {total_latency:.4f} seconds')
print(f'Average latency per frame: {average_latency_per_frame_ms:.4f} milliseconds')
print(f'Throughput: {throughput:.4f} frames/second')

cv2.destroyAllWindows()
