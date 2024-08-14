import csv
import cv2
from ultralytics import YOLO
import numpy as np

VIDEO_FILE = 'DJI_0886.MP4'
MODEL_PATH = 'yolov8n-pose.pt'
CSV_FILE = 'keypoints.csv'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(VIDEO_FILE)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create CSV file and write header
fieldnames = [
    'Nose', 'Left Wrist', 'Right Wrist', 'Left Elbow', 'Right Elbow',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Elbow Angle', 
    'Right Elbow Angle', 'Left Knee Angle', 'Right Knee Angle'
]
with open(CSV_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fieldnames)

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points: point1 (A), point2 (B), point3 (C).
    The angle is calculated at point2 (B).
    """
    a = np.array(point1)  # first segment
    b = np.array(point2)  # joint
    c = np.array(point3)  # second segment
    
    # Vectors between the points
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using the dot product formula
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clip the cosine value to ensure it lies between -1 and 1 to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert it to degrees
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def write_pose_video(video_file, csv_file, frame_lock, current_frame):
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with YOLOv8
        results = model(frame)
        if results:
            result = results[0]
            if hasattr(result, 'keypoints') and result.keypoints.data.numel() > 0:
                keypoints = result.keypoints.data.cpu().numpy()[0]

                # Extract keypoint positions
                nose = keypoints[0]
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]
                left_elbow = keypoints[7]
                right_elbow = keypoints[8]
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]

                # Calculate angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Append results to the CSV file
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        nose[1], left_wrist[1], right_wrist[1], left_elbow[1], right_elbow[1], 
                        left_hip[1], right_hip[1], left_knee[1], right_knee[1], 
                        left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle
                    ])

        with frame_lock:
            current_frame[0] = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
