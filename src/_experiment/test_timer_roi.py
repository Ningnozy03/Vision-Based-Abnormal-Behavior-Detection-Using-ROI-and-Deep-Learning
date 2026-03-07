import cv2
import time

roi = None
drawing = False
x1, y1 = 0, 0

# จุดเมาส์ (แทน object)
mouse_x, mouse_y = -1, -1

ABNORMAL_SECONDS = 30.0
inside_start_time = None
triggered = False

def normalize_roi(r):
    x1, y1, x2, y2 = r
    return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)

def point_in_roi(px, py, r):
    rx1, ry1, rx2, ry2 = r
    return rx1 <= px <= rx2 and ry1 <= py <= ry2

def on_mouse(event, x, y, flags, param):
    global roi, drawing, x1, y1, mouse_x, mouse_y

    # ตำแหน่งเมาส์ตลอดเวลา
    mouse_x, mouse_y = x, y

    # ลาก ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = normalize_roi((x1, y1, x, y))
        print("✅ ROI:", roi)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    raise SystemExit

cv2.namedWindow("Timer ROI Test")
cv2.setMouseCallback("Timer ROI Test", on_mouse)

print("✅ ลาก ROI ด้วยเมาส์ซ้าย | เอาเมาส์ไปไว้ใน ROI เพื่อจำลอง object | กด ESC ออก")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ อ่านเฟรมไม่ได้")
        break

    # วาดจุดเมาส์ (แทน object)
    if mouse_x != -1:
        cv2.circle(frame, (mouse_x, mouse_y), 8, (255, 255, 255), -1)

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        inside = point_in_roi(mouse_x, mouse_y, roi)

        if inside:
            if inside_start_time is None:
                inside_start_time = time.time()
                triggered = False

            elapsed = time.time() - inside_start_time

            # แสดงเวลาที่อยู่ใน ROI
            cv2.putText(frame, f"IN ROI: {elapsed:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if elapsed >= ABNORMAL_SECONDS and not triggered:
                triggered = True

            if triggered:
                cv2.putText(frame, "ABNORMAL (>=30s)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        else:
            inside_start_time = None
            triggered = False
            cv2.putText(frame, "OUT ROI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "Draw ROI first", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Timer ROI Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
