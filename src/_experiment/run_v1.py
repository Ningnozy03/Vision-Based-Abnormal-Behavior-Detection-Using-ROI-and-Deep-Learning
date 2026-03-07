import cv2
from ultralytics import YOLO

from roi_io import load_roi, save_roi
from event_manager import EventManager
from event_logger import save_event_image, log_event, ensure_csv

ROI_PATH = "configs/roi.json"
CSV_PATH = "outputs/logs/events.csv"
IMG_DIR = "outputs/events"

ABNORMAL_SECONDS = 30.0
RESET_MISSING_SECONDS = 2.0
CONF_THRES = 0.4

# COCO: person=0, car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASS_IDS = {0, 2, 3, 5, 7}

roi = None
drawing = False
x1, y1 = 0, 0

def normalize_roi(r):
    x1, y1, x2, y2 = r
    return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)

def center_of_bbox(b):
    x1, y1, x2, y2 = b
    return int((x1+x2)/2), int((y1+y2)/2)

def point_in_roi(px, py, r):
    return r["x1"] <= px <= r["x2"] and r["y1"] <= py <= r["y2"]

def on_mouse(event, x, y, flags, param):
    global roi, drawing, x1, y1
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx1, ry1, rx2, ry2 = normalize_roi((x1, y1, x, y))
        roi = {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2}
        save_roi(ROI_PATH, roi)
        print("✅ ROI saved:", roi)

def main():
    global roi

    ensure_csv(CSV_PATH)

    roi = load_roi(ROI_PATH)
    if roi:
        print("✅ ROI loaded:", roi)
    else:
        print("ℹ️ ยังไม่มี ROI — ลากกรอบเพื่อบันทึก")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ เปิดกล้องไม่ได้")
        return

    cv2.namedWindow("Run V1")
    cv2.setMouseCallback("Run V1", on_mouse)

    em = EventManager(ABNORMAL_SECONDS, RESET_MISSING_SECONDS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw ROI
        if roi:
            cv2.rectangle(frame, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), (0, 255, 255), 2)
            cv2.putText(frame, "WARNING ROI", (roi["x1"], max(20, roi["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Draw ROI to save (mouse drag)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # detect+track
        results = model.track(frame, persist=True, conf=CONF_THRES, verbose=False)

        keys_in_roi = set()
        key_to_cls = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            clss = boxes.cls
            confs = boxes.conf
            xyxys = boxes.xyxy

            for i in range(len(xyxys)):
                cls_id = int(clss[i].item())
                if cls_id not in TARGET_CLASS_IDS:
                    continue

                conf = float(confs[i].item())
                x1, y1, x2, y2 = map(int, xyxys[i].tolist())
                cx, cy = center_of_bbox((x1, y1, x2, y2))

                tid = None
                if ids is not None:
                    tid = int(ids[i].item())

                key = tid if tid is not None else (cls_id, x1, y1, x2, y2)
                key_to_cls[key] = cls_id

                # draw bbox
                label = f"ID:{tid} cls:{cls_id} {conf:.2f}" if tid is not None else f"cls:{cls_id} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if roi and point_in_roi(cx, cy, roi):
                    keys_in_roi.add(key)
                    elapsed = em.get_elapsed(key)
                    cv2.putText(frame, f"{elapsed:.1f}s", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if em.is_triggered(key):
                        cv2.putText(frame, "ABNORMAL", (x1, y2 + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

        # update events
        events = em.update(keys_in_roi, key_to_cls)
        for ev in events:
            # save evidence
            img_path = save_event_image(IMG_DIR, frame, ev["key"], ev["cls"])
            log_event(CSV_PATH, ev["key"], ev["cls"], ev["elapsed"], img_path)
            print(f"🚨 EVENT saved | key={ev['key']} cls={ev['cls']} elapsed={ev['elapsed']:.1f}s")

        cv2.imshow("Run V1", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
