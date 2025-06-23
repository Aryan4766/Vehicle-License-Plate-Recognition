from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt if needed

def detect_vehicles(frame):
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in [2, 3, 5, 7]:  # Only keep vehicle classes (car, motorbike, bus, truck)
                detections.append((x1, y1, x2, y2, cls, conf))

    return detections

