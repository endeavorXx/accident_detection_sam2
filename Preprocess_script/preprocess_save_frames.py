import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')  # you can use 'yolov8s.pt', 'yolov8m.pt', etc.

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to check proximity
def check_proximity(boxes, threshold=50):
    close_pairs = []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            center_i = ((boxes[i][0] + boxes[i][2]) // 2, (boxes[i][1] + boxes[i][3]) // 2)
            center_j = ((boxes[j][0] + boxes[j][2]) // 2, (boxes[j][1] + boxes[j][3]) // 2)
            distance = euclidean_distance(center_i, center_j)
            if distance < threshold:
                close_pairs.append((i, j))
    return close_pairs

# Path to the directory containing the frames
frames_dir = '/media/sgsharma/New Volume/midas/accident_detection/DoTA/frames'
output_dir = '/media/sgsharma/New Volume/midas/accident_detection/DoTA/processed_frames'  # Set your desired output directory

# List all video folders
video_folders = glob.glob(os.path.join(frames_dir, '*'))

# Limit to only 5 video folders
video_folders = video_folders[:5]

# Process each video folder
for video_folder in tqdm(video_folders, desc='Processing video folders'):
    images_dir = os.path.join(video_folder, 'images')
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))

    # Create output directory for the processed frames
    output_video_dir = os.path.join(output_dir, os.path.basename(video_folder))
    os.makedirs(output_video_dir, exist_ok=True)

    for image_file in tqdm(image_files, desc=f'Processing frames in {os.path.basename(video_folder)}', leave=False):
        frame = cv2.imread(image_file)

        # Perform object detection
        results = model(frame)

        # Extract bounding boxes and labels
        boxes = []
        for result in results[0].boxes:
            if result.cls == 2:  # Class 2 is 'car' in COCO dataset
                x1, y1, x2, y2 = result.xyxy[0].tolist()  # Convert tensor to list
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Check proximity
        close_pairs = check_proximity(boxes)

        # Draw bounding boxes and proximity alerts
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = (0, 255, 0) if all(i not in pair for pair in close_pairs) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Save the processed frame
        output_file = os.path.join(output_video_dir, os.path.basename(image_file))
        cv2.imwrite(output_file, frame)

cv2.destroyAllWindows()