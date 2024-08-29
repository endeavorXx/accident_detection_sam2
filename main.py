import os
import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from yolofunctions import *
from preprocess import *



video_path = r"C:\Users\Asus\Desktop\Accident_temp\inputs\001.mp4"
#video_path = r"C:\Users\akhil\Downloads\WhatsApp Video 2024-05-24 at 7.20.57 AM.mp4"
video_out_path = os.path.join('.', 'out_vid.mp4')

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov10m.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

detection_threshold = 0.4

trajectories = {}

near_accidents = {}
potential_accidents = {}

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 5)  # 5 seconds interval

final_ll = []

total_latency = 0
detection_latency = 0
tracking_latency = 0

frames=0

while ret:
    start_time = time.time()
    detection_start_time = time.time()
    results = model(frame)
    detection_end_time = time.time()
    detection_latency += (detection_end_time - detection_start_time)

    detections = []

    proximity_flag,_,_=classical_process(frame)
    print(proximity_flag)
    if(True):
        print("here")
        frames+=1
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

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if track_id in trajectories:
                    del trajectories[track_id]

        # Remove trajectories and near-accident messages for vehicles no longer in frame
        visible_track_ids = {track.track_id for track in tracker.tracks}
        trajectories = {track_id: traj for track_id, traj in trajectories.items() if track_id in visible_track_ids}
        near_accidents = {key: expire_time for key, expire_time in near_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}
        potential_accidents = {key: expire_time for key, expire_time in potential_accidents.items() if key[0] in visible_track_ids and key[1] in visible_track_ids}

        # Draw trajectories and annotate distances between different vehicles
        track_ids = list(trajectories.keys())
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                t_id1, t_id2 = track_ids[i], track_ids[j]
                traj1, traj2 = trajectories[t_id1], trajectories[t_id2]
                if traj1 and traj2:
                    pt1, pt2 = traj1[-1], traj2[-1]  # Get the latest points (centroids)
                    distance = calculate_distance(pt1, pt2)
                    #cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
                    midpoint = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
                    cv2.putText(frame, f'{distance:.2f}', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect and label near-accidents and potential accidents
        current_time = frame_count / fps
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
                    if calculate_iou(bbox1, bbox2) > 0.1:
                        potential_accidents[(t_id1, t_id2)] = current_time + 5
                    if detect_near_accident(traj1, traj2, bbox1, bbox2) and abs(direction1 - direction2) < np.pi / 4:
                        # Store near accident message with expiration time
                        near_accidents[(t_id1, t_id2)] = current_time + 5

        # Draw potential accident messages and remove expired ones
        expired_keys = []
        for (t_id1, t_id2), expire_time in potential_accidents.items():
            if current_time > expire_time:
                expired_keys.append((t_id1, t_id2))
            else:
                # Get the last known positions of the tracks
                if t_id1 in trajectories and t_id2 in trajectories:
                    pt1 = trajectories[t_id1][-1]
                    pt2 = trajectories[t_id2][-1]
                    midpoint = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
                    # cv2.putText(frame, "Potential Accident", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Remove expired potential accident messages
        for key in expired_keys:
            del potential_accidents[key]

        # Draw near accident messages and remove expired ones
        expired_keys = []
        for (t_id1, t_id2), expire_time in near_accidents.items():
            if current_time > expire_time:
                expired_keys.append((t_id1, t_id2))
            else:
                # Get the last known positions of the tracks
                if t_id1 in trajectories and t_id2 in trajectories:
                    pt1 = trajectories[t_id1][-1]
                    pt2 = trajectories[t_id2][-1]
                    midpoint = (int((pt1[0] + pt2[0]) / 2) - 3, int((pt1[1] + pt2[1]) / 2))
                    cv2.putText(frame, "Near Accident!", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Remove expired near-accident messages
        for key in expired_keys:
            del near_accidents[key]

        cap_out.write(frame)

        # Save trajectories on a white background every 5 seconds
        if frame_count % interval_frames == 0:
            white_board = np.ones_like(frame) * 255  # Create a white background
            for track_id, points in trajectories.items():
                if len(points) > 1:
                    # Write the ID at the starting point of the trajectory
                    start_pt = (int(points[0][0]), int(points[0][1]))
                    cv2.putText(white_board, f'ID: {track_id}', start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Draw the trajectory lines
                    for i in range(1, len(points)):
                        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                        pt2 = (int(points[i][0]), int(points[i][1]))
                        cv2.line(white_board, pt1, pt2, colors[track_id % len(colors)], 2)
            cv2.imwrite(f'frame_{frame_count}.png', white_board)

    frame_count += 1
    ret, frame = cap.read()
    
    end_time = time.time()
    total_latency += (end_time - start_time)

cap.release()
cap_out.release()
cv2.destroyAllWindows()

# Calculate average latencies
avg_detection_latency = detection_latency / frame_count
avg_tracking_latency = tracking_latency / frame_count
avg_total_latency = total_latency / frame_count

print("Average Detection Latency per Frame: {:.4f} seconds".format(avg_detection_latency))
print("Average Tracking Latency per Frame: {:.4f} seconds".format(avg_tracking_latency))
print("Average Total Latency per Frame: {:.4f} seconds".format(avg_total_latency))

print(frames)
