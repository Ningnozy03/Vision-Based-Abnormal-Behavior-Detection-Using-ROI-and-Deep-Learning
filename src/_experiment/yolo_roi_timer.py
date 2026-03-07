import cv2
import time
from ultralytics import YOLO

# ========= CONFIG =========
ABNORMAL_SECONDS = 30.0
CONF_THRES = 0.4

# YOLO classes (COCO): person=0, car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASS_IDS = {0, 2, 3, 5, 7}

roi = None
drawing = False
x1, y1 = 0, 0

# per track-id state (เราจะใช้ "id จาก tracker ของ YOLO" ถ้าได้)
state = {}  # id -> {"start_time": float, "last_seen": float, "triggered": bool, "cls": str}

def normalize_roi(r):
    x1, y1, x2, y2 = r
    return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)

def center_of_bbox(b):
    x1, y1, x2, y2 = b
    return int((x1+x2)/2), int((y1+y2)/2)

def point_in_roi(px, py, r):
    rx1, ry1, rx2, ry2 = r
    return rx1 <= px <= rx2 and ry1 <= py <= ry2

def on_mouse(event, x, y, flags, param):
    global roi, drawing, x1, y1
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = normalize_roi((x1, y1, x, y))
        print("✅ ROI:", roi)

# ========= LOAD MODEL =========
model = YOLO("yolov8n.pt")  # รุ่นเล็ก โหลดไว เหมาะเริ่มต้น

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    raise SystemExit

cv2.namedWindow("YOLO ROI Timer")
cv2.setMouseCallback("YOLO ROI Timer", on_mouse)

print("✅ ลาก ROI ก่อน แล้วให้คน/รถเข้า ROI ค้าง ≥ 30 วิ เพื่อ trigger")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # วาด ROI
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(frame, "WARNING ROI", (rx1, max(20, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Draw ROI first", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ===== DETECT + TRACK =====
    # persist=True จะช่วยให้ model เก็บ track id ต่อเนื่อง (ถ้า tracker พร้อม)
    results = model.track(frame, persist=True, conf=CONF_THRES, verbose=False)

    now = time.time()

    # เก็บ id ที่เห็นใน ROI รอบนี้
    seen_in_roi = set()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        # บางเครื่อง boxes.id อาจเป็น None (tracker ยังไม่ทำงาน) -> fallback ใช้ None id
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

            # id อาจไม่มี
            tid = None
            if ids is not None:
                tid = int(ids[i].item())

            # วาด bbox
            label = f"cls:{cls_id} {conf:.2f}"
            if tid is not None:
                label = f"ID:{tid} " + label

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ถ้ายังไม่ได้ลาก ROI ก็ข้าม logic เวลา
            if roi is None:
                continue

            inside = point_in_roi(cx, cy, roi)
            if not inside:
                continue

            # ==== IN ROI ====
            # ถ้าไม่มี tid ให้สร้าง key แบบชั่วคราวจาก bbox (ไม่ดีเท่า id แต่พอเริ่มได้)
            key = tid if tid is not None else (cls_id, x1, y1, x2, y2)

            seen_in_roi.add(key)

            if key not in state:
                state[key] = {"start_time": now, "last_seen": now, "triggered": False, "cls": str(cls_id)}
            else:
                state[key]["last_seen"] = now

            elapsed = now - state[key]["start_time"]

            # แสดงเวลาบน bbox
            cv2.putText(frame, f"{elapsed:.1f}s", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if (not state[key]["triggered"]) and elapsed >= ABNORMAL_SECONDS:
                state[key]["triggered"] = True
                print(f"🚨 ABNORMAL! key={key} elapsed={elapsed:.1f}s")

            if state[key]["triggered"]:
                cv2.putText(frame, "ABNORMAL", (x1, y2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    # ===== RESET คนที่หายไปจาก ROI นานเกิน 2 วิ =====
    to_delete = []
    for k, v in state.items():
        if k not in seen_in_roi:
            if (now - v["last_seen"]) > 2.0:
                to_delete.append(k)
    for k in to_delete:
        del state[k]

    cv2.imshow("YOLO ROI Timer", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
