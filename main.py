import cv2
import os
import sys
import threading
import pandas as pd
from datetime import datetime

# Fix import path for subfolders
sys.path.append(os.path.join(os.path.dirname(__file__), 'detectors'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from vehicle_detector import detect_vehicles
from license_plate_reader import read_license_plate
from face_detector import detect_faces
from color_utils import get_dominant_color

# Video paths (set full paths here)
video_paths = {
    "Cam1": r"data\sample_cam1.mp4",
    "Cam2": r"data\sample_cam2.mp4",
    "Cam3": r"data\sample_cam3.mp4"
}

# Setup log
log_data = []
os.makedirs("logs/images", exist_ok=True)

def process_video(cam_name, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    print(f"[INFO] Starting {cam_name} - {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"[INFO] Processing {cam_name} - Frame {frame_id}")

        detections = detect_vehicles(frame)
        print(f"[INFO] Detected {len(detections)} vehicles in {cam_name}, Frame {frame_id}")

        for i, (x1, y1, x2, y2, cls, conf) in enumerate(detections):
            obj_img = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")
            plate = read_license_plate(obj_img)
            color = get_dominant_color(obj_img)
            image_name = f"{cam_name.lower()}_frame{frame_id}_obj{i}.jpg"
            image_path = os.path.join("logs/images", image_name)
            cv2.imwrite(image_path, obj_img)

            print(f"[✓] {cam_name} Frame {frame_id} Obj {i}: Plate={plate} Color={color} saved at {image_path}")

            log_data.append({
                "Camera": cam_name,
                "Frame": frame_id,
                "Timestamp": timestamp,
                "Class": cls,
                "Confidence": round(conf, 2),
                "LicensePlate": plate,
                "Color": color,
                "ImagePath": image_path
            })

        # ✅ Save CSV after each frame
        pd.DataFrame(log_data).to_csv("logs/detections.csv", index=False)

    cap.release()

# Start threads for each camera
threads = []
for cam_name, video_path in video_paths.items():
    t = threading.Thread(target=process_video, args=(cam_name, video_path))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("[✅] Detection CSV saved to logs/detections.csv")

