import csv
import cv2
from ultralytics import YOLO

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
fieldnames = ['Nose', 'Left Wrist', 'Right Wrist', 'Left Elbow', 'Right Elbow', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']
with open(CSV_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fieldnames)

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
                keypoints = result.keypoints.data
                # Extract only the x-coordinates for the specified keypoints
                keypoint_values = [
                    int(keypoints[0][0][0] * width),   # Nose
                    int(keypoints[0][9][0] * width),   # Left Wrist
                    int(keypoints[0][10][0] * width),  # Right Wrist
                    int(keypoints[0][7][0] * width),   # Left Elbow
                    int(keypoints[0][8][0] * width),   # Right Elbow
                    int(keypoints[0][11][0] * width),  # Left Hip
                    int(keypoints[0][12][0] * width),  # Right Hip
                    int(keypoints[0][13][0] * width),  # Left Knee
                    int(keypoints[0][14][0] * width)   # Right Knee
                ]

                # Append results to the CSV file
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(keypoint_values)

        with frame_lock:
            current_frame[0] = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
