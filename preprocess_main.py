
import cv2
import os
import preprocess

import numpy as np

def process_frame(frame):
    # Example processing: convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def process_video(input_path, output_path):

    # Reset background subtractor
    preprocess.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = None
    target_fps = 10

    full_frame_shape = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, c = frame.shape
        flag, metadata = preprocess.is_critical(frame)
        
        # Constuct full frame using metadata
        if full_frame_shape == None:
            full_frame_shape = (h*2, w*2, c)

        full_frame = np.zeros(full_frame_shape) 
        full_frame += 255
        # Add misc data to full_frame
        full_frame[0:h, 0:w, :] = frame
        full_frame[h:, 0:w, :] = metadata["yolo"]
        full_frame[0:h, w:, :] = metadata["classical"]
        full_frame[h:, w:, :] = metadata["classical2"]

        full_frame = full_frame.astype('uint8')

        # Write to output
        if out == None:
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (full_frame_shape[1], full_frame_shape[0]), isColor=True)
        
        # print(frame.max(), frame.min())
        # print(metadata["yolo"].max(), metadata["yolo"].min())
        print(full_frame.shape, full_frame.dtype)
        out.write(full_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print("\nProcessing complete!")

if __name__ == "__main__":

    if "outputs" not in os.listdir():
        os.mkdir("outputs")

    inputs = os.listdir("inputs")

    for fname in inputs:
        name = fname.split('.')[0]
        input_path = os.path.join("inputs", fname)
        output_path = os.path.join("outputs", fname)
        process_video(input_path, output_path)
