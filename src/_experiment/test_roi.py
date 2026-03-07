import cv2

roi = None
drawing = False
x1, y1 = 0, 0

def draw_roi(event, x, y, flags, param):
    global roi, drawing, x1, y1

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        # normalize ให้เป็นซ้ายบน-ขวาล่าง
        rx1, ry1 = min(x1, x2), min(y1, y2)
        rx2, ry2 = max(x1, x2), max(y1, y2)
        roi = (rx1, ry1, rx2, ry2)
        print("✅ ROI:", roi)

# ลองใช้ DSHOW บน Windows จะเสถียรกว่า
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้ (ลองเปลี่ยนเป็น 1 หรือปิดแอปที่ใช้กล้องอยู่ เช่น Zoom/Line/Camera)")
    raise SystemExit

cv2.namedWindow("ROI Test")
cv2.setMouseCallback("ROI Test", draw_roi)

print("✅ กล้องเปิดแล้ว (กด ESC เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ อ่านเฟรมไม่ได้")
        break

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(frame, "Drag ROI (ESC to exit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ROI Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
