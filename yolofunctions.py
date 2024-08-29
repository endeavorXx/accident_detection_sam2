import numpy as np

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
    #final_ll.append(np.linalg.norm(np.array(traj1[-1]) - np.array(traj2[-1])))
    iou_condition = calculate_iou(bbox1, bbox2) > iou_threshold
    return distance_condition or iou_condition

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2,y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union area
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
