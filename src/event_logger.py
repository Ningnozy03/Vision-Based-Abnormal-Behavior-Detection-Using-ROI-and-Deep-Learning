import csv
import os
import time
import cv2

def ensure_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "track_key", "cls", "duration_sec", "image_path"])

def save_event_image(img_dir: str, frame, track_key, cls):
    os.makedirs(img_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_key{str(track_key).replace(' ','')}_{cls}.jpg"
    path = os.path.join(img_dir, fname)
    cv2.imwrite(path, frame)
    return path

def log_event(csv_path: str, track_key, cls, duration_sec: float, image_path: str):
    ensure_csv(csv_path)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([ts, track_key, cls, f"{duration_sec:.2f}", image_path])
