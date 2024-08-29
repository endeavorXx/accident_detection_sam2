import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, fps=10):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the original fps of the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval between frames to capture
    frame_interval = int(video_fps / fps)
    if frame_interval == 0:
        return
    
    frame_count = 0
    saved_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it's at the correct interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()

def process_videos_in_folder(input_folder, output_root_folder, fps=10):
    for root, _, files in os.walk(input_folder):
        print(f"processing files in folder: {root}")
        for file in tqdm(files, desc = "files processing"):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_folder = os.path.join(output_root_folder, relative_path, os.path.splitext(file)[0])
                extract_frames(video_path, output_folder, fps)

if __name__ == "__main__":
    input_folder = '/media/sgsharma/New Volume/midas/accident_detection/IEEE_dataset/orig_videos/train'
    output_root_folder = '/media/sgsharma/New Volume/midas/accident_detection/IEEE_dataset/processed_frames/train'
    
    process_videos_in_folder(input_folder, output_root_folder, fps=10)
