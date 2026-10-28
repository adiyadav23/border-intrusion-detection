import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
import time

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = "video.mp4"   # 0 for webcam
CONFIDENCE = 0.5
BORDER_Y = 300
SAVE_DIR = "evidence"
FRAME_SKIP = 3               # higher = faster
INPUT_SIZE = (640, 384)     # smaller = faster

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_SOURCE)

frame_count = 0
last_saved_time = 0

# ================= GREEN DETECTION =================
def is_green_object(frame, x1, y1, x2, y2):
    # Downsample ROI for speed
    roi = frame[y1:y2:2, x1:x2:2]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.mean(mask > 0)

    return green_ratio > 0.3

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Resize for speed
    frame = cv2.resize(frame, INPUT_SIZE)

    # YOLO inference (ONLY cars)
    results = model(frame, classes=[2], conf=CONFIDENCE, verbose=False)

    intrusion_detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Ignore green cars
            if is_green_object(frame, x1, y1, x2, y2):
                continue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "CAR", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Intrusion logic
            if y2 > BORDER_Y:
                intrusion_detected = True
                cv2.putText(frame, "CAR INTRUSION!", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw border line
    cv2.line(frame, (0, BORDER_Y), (frame.shape[1], BORDER_Y), (255, 0, 0), 2)

    # Save evidence (2 sec cooldown)
    if intrusion_detected:
        current_time = time.time()
        if current_time - last_saved_time > 2:
            filename = os.path.join(
                SAVE_DIR,
                f"car_intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(filename, frame)
            print("Car intrusion saved:", filename)
            last_saved_time = current_time

    # Show output
    cv2.imshow("Intrusion Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
