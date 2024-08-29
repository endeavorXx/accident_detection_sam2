import cv2
import os

def extract_latest_frames(video_path, output_folder, max_frames=200):
    print("Processing........")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    start_frame = max(0, total_frames - max_frames)
    frame_indices = range(start_frame, total_frames)
    
    frame_count = 0
    saved_frames = 0
    while True:
        success, frame = video_capture.read()
        if not success or saved_frames >= max_frames:
            break
        
        if frame_count in frame_indices:
            output_path = os.path.join(video_output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frames += 1
        
        frame_count += 1

    video_capture.release()

def process_videos(input_folder, output_folder):
    # Supported video formats
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    for filename in os.listdir(input_folder):
        if filename.endswith(video_extensions):
            video_path = os.path.join(input_folder, filename)
            extract_latest_frames(video_path, output_folder)



input_folder = r"C:\Users\akhil\Akhil_work\test"
output_folder = r'Accident_output_frames_CCTV'

'''for dir_name in os.listdir(input_folder):
    dir_path = os.path.join(input_folder, dir_name)
    if os.path.isdir(dir_path):
        process_videos(dir_path, output_folder)'''


# Process all videos in the input folder




