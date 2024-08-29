import os
import cv2

frames_dir = r"C:\Users\akhil\OneDrive\Desktop\Test_perspective_divided\Test\CCTV"
accident_output_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Accident_output_frames"
non_accident_output_folder = r"C:\Users\akhil\Akhil_work\Baseline\Output\CCTV_Non_Accident_output_frames"

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success:
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        success, frame = cap.read()
        count += 1
    cap.release()

def process_videos(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video)
        if video_path.endswith(".mp4"):
            video_output_folder = os.path.join(output_folder, os.path.splitext(video)[0])
            extract_frames(video_path, video_output_folder)

process_videos(os.path.join(frames_dir, "Accident"), accident_output_folder)
process_videos(os.path.join(frames_dir, "Non accident"), non_accident_output_folder)
