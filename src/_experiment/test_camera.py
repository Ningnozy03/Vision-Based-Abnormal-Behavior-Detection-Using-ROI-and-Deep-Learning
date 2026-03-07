import cv2

cap = cv2.VideoCapture(0)  # 0 = กล้องตัวแรก

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ อ่านภาพไม่ได้")
        break

    cv2.imshow("My Camera", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
