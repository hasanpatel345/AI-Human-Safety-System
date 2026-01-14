import cv2
from ultralytics import YOLO
import winsound
import time
import json
import csv
import os
from datetime import datetime

# -------------------------------------------------
# Resolve base directory (PROJECT ROOT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------
# Load configuration
# -------------------------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

CAMERA_INDEX = config["camera_index"]
ZONE = config["restricted_zone"]
ALERT_FREQ = config["alert"]["sound_frequency"]
ALERT_DURATION = config["alert"]["sound_duration"]
ALERT_DELAY = config["alert"]["cooldown_seconds"]
LOG_FILE = os.path.join(BASE_DIR, config["log_file"])

# -------------------------------------------------
# Initialize CSV log file safely
# -------------------------------------------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "Confidence"])

# -------------------------------------------------
# Load YOLOv8 model
# -------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model", "yolov8n.pt")
model = YOLO(MODEL_PATH)

# -------------------------------------------------
# Open webcam
# -------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ ERROR: Unable to access camera")
    exit()

last_alert_time = 0

# -------------------------------------------------
# Main loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read frame")
        break

    results = model(frame, verbose=False)

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (ZONE["x1"], ZONE["y1"]),
        (ZONE["x2"], ZONE["y2"]),
        (0, 0, 255),
        2
    )
    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (ZONE["x1"], ZONE["y1"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"HUMAN {confidence:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Center point calculation
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # Zone violation check
                inside_zone = (
                    ZONE["x1"] < center_x < ZONE["x2"] and
                    ZONE["y1"] < center_y < ZONE["y2"]
                )

                if inside_zone:
                    cv2.putText(
                        frame,
                        "ALERT! SAFETY VIOLATION",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    current_time = time.time()
                    if current_time - last_alert_time > ALERT_DELAY:
                        winsound.Beep(ALERT_FREQ, ALERT_DURATION)
                        last_alert_time = current_time

                        # Write log entry
                        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Human entered restricted zone",
                                round(confidence, 2)
                            ])

    cv2.imshow("AI Human Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()
