
import cv2
import os, sys, random, time
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO
import numpy as np


yolo = YOLO("yolov8n.pt")

# Hard thresholds
CONF_THRESHOLD = 0.4
NEARBY_THRESHOLD = 40 # Pixel distance
BYPASS_THRESHOLD = 30

########
# YOLO #
########

is_overlapping = lambda a1,a2,b1,b2: (a2 >= b1) and (b2 >= a1)

def draw_bounding_boxes(image, preds):
    for pred in preds:
        x1, y1, x2, y2, conf, label = pred
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Car: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def yolo_process(image):

    # # Get number of cars in this image
    # results = yolo(image)
    # preds = results[0].boxes.data # x1, y1, x2, y2, conf, label
    # preds = preds[preds[:, 4] > CONF_THRESHOLD]
    # if len(preds) <= 1: return False, None # Assuming this system is being used for CCTV Cams and not Dashcams
    # if len(preds) >= BYPASS_THRESHOLD: return True, preds
    # # Collect all useful predictions
    # n, k = len(preds), NEARBY_THRESHOLD
    # for i in range(n):
    #     for j in range(i+1, n):
    #         # Check if i and j overlap
    #         a, b = preds[i], preds[j]
    #         flag = is_overlapping(a[0] - k, a[2] + k, b[0], b[2]) and is_overlapping(a[1] - k, a[3] + k, b[1], b[3])
    #         if flag: return True, preds
    # return False, None

    # Get number of cars in this image
    results = yolo(image)
    preds = results[0].boxes.data # x1, y1, x2, y2, conf, label
    preds = preds[preds[:, 4] > CONF_THRESHOLD]

    if len(preds) <= 1:
        return False, image # No need to draw boxes if there are 0 or 1 car

    if len(preds) >= BYPASS_THRESHOLD:
        draw_bounding_boxes(image, preds)
        return True, image

    # Collect all useful predictions
    n, k = len(preds), NEARBY_THRESHOLD
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j overlap
            a, b = preds[i], preds[j]
            flag = is_overlapping(a[0] - k, a[2] + k, b[0], b[2]) and is_overlapping(a[1] - k, a[3] + k, b[1], b[3])
            if flag:
                draw_bounding_boxes(image, preds)
                return True, image

    draw_bounding_boxes(image, preds)
    return False, image

#############
# Classical #
#############

def check_proximity(boxes, threshold=NEARBY_THRESHOLD):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[j]
            flag = (is_overlapping(x1 - threshold, x1 + w1 + threshold, x2, x2 + w2) and
                    is_overlapping(y1 - threshold, y1 + h1 + threshold, y2, y2 + h2))
            if flag:
                return True
    return False

# Initialize the background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
def classical_process(image):
    # Apply background subtraction
    fg_mask = back_sub.apply(image)

    # Thresholding to binarize the input to 0/1
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Apply Gaussian Blur to smooth the image
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Use morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

    ## Seperate out nearby objects, Fil holes 
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel) # Dilation -> Erosion
    ## Eliminate noise and small objects
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) # Erosion -> Dilation

    # Find contours in the foreground mask
    ## RETR_External: Only extract the outer contours of an object
    ## Only store edges of the contours, more memory efficient
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    foreground = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes around detected vehicles
    for contour in contours:
        # Estimate the area of the contour
        area = cv2.contourArea(contour)
        if area > 800:  # Filter out small contours
            # Approximate the contour to a bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the bounding box resembles the aspect ratio of a vehicle
            aspect_ratio = float(w) / h
            if 0.2 <= aspect_ratio <= 4.0:  # Filter out non-vehicle shapes based on aspect ratio
                # Append the bounding box coordinates to the list
                boxes.append((x, y, w, h))
                # Draw bounding box on the original image and foreground mask
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(foreground, (x, y), (x + w, y + h), (0, 255, 0), 2)

    proximity_flag = check_proximity(boxes)

    return proximity_flag, foreground, image


def add_border(image, flag, border_thickness=15):
    if flag:
        # Add a green border around the image within the existing shape
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, height), (0, 255, 0), border_thickness)
    return image

def put_text(image, text, position=(20, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def is_critical(frame):

    metadata = {}

    # Yolo
    yolo_frame = np.copy(frame)
    flag, annotated_image = yolo_process(np.copy(yolo_frame))
    annotated_image = add_border(annotated_image, flag)
    metadata["yolo"] = annotated_image

    # Classical CV
    classical_frame = np.copy(frame)
    flag, foreground, annotated_image = classical_process(classical_frame)
    annotated_image = add_border(annotated_image, flag)
    metadata["classical"] = foreground
    metadata["classical2"] = annotated_image

    metadata["yolo"] = put_text(metadata["yolo"], "YOLO")
    metadata["classical"] = put_text(metadata["classical"], "FOREGROUND")
    metadata["classical2"] = put_text(metadata["classical2"], "CLASSICAL CV")

    return flag, metadata

