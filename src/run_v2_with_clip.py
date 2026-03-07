import os
import cv2
import time
import csv
import json
import collections
import subprocess
from ultralytics import YOLO

try:
    # run as module: python -m src.run_v2_with_clip
    from .shared_state import status, recent_events, latest_jpg_lock
    import src.shared_state as shared_state
except ImportError:
    # run as script: python src/run_v2_with_clip.py
    from shared_state import status, recent_events, latest_jpg_lock
    import shared_state

# ========================================================================================
# ===== RESOLUTION =====
# Downscale for YOLO inference, Upscale for display / streaming
INFER_W, INFER_H = 512, 288
DISP_W,  DISP_H  = 960, 540

# ===== CONFIG =====
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

# For testing, I can use a local video file or My webcam by changing the url variable.
url = "rtsp://Ningfinal:20032005@192.168.0.110:554/stream1"

ROI_PATH = "configs/roi.json"
CONFIG_PATH = "configs/config.json"
EVENT_DIR = "outputs/events"
CLIP_DIR = "outputs/clips"
LOG_DIR = "outputs/logs"
LIVE_DIR = "outputs/live"
LATEST_JPG_PATH = os.path.join(LIVE_DIR, "latest.jpg")

os.makedirs(LIVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, "events.csv")
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.csv")
ROI_ID = "roi_1"

DEFAULT_CONFIG = {
    "abnormal_seconds": 30.0,     # 10/30/60
    "roi_enabled": True,          # toggle ROI logic
    "conf_thres": 0.4,            # yolo confidence threshold
    "pre_seconds": 10.0,          # pre buffer seconds
    "post_seconds": 5.0           # post record seconds
}
RESET_MISSING_SECONDS = 2.0

TARGET_CLASS_IDS = {0, 2, 3, 5, 7}  
CLS_NAME = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# =========================
# CONFIG IO
# =========================
def load_config(path=CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
    except Exception:
        cfg = {}

    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    return merged

def save_config(cfg, path=CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# =========================
# ROI IO
# =========================
def load_roi(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_roi(path, roi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(roi, f, ensure_ascii=False, indent=2)

def normalize_roi(x1, y1, x2, y2):
    return {"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)}

def center_of_bbox(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def in_roi(cx, cy, roi):
    return roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]

# =========================
# CSV Logging
# =========================
def ensure_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "roi_id",
                "track_key",
                "class",
                "elapsed_sec",
                "snapshot_path",
                "clip_path",
                "source"
            ])

def append_event_csv(csv_path: str, roi_id, track_key, cls_name, elapsed_sec, snapshot_path, clip_path, source):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            ts,
            roi_id,
            track_key,
            cls_name,
            f"{elapsed_sec:.2f}",
            snapshot_path,
            clip_path,
            source
        ])

def update_summary(summary_path: str, cls_name: str):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    today = time.strftime("%Y-%m-%d")

    rows = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows[row["date"]] = row

    if today not in rows:
        rows[today] = {"date": today, "person_events": "0", "vehicle_events": "0", "total_events": "0"}

    person = int(rows[today]["person_events"])
    vehicle = int(rows[today]["vehicle_events"])
    total = int(rows[today]["total_events"])

    if cls_name == "person":
        person += 1
    else:
        vehicle += 1
    total += 1

    rows[today]["person_events"] = str(person)
    rows[today]["vehicle_events"] = str(vehicle)
    rows[today]["total_events"] = str(total)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "person_events", "vehicle_events", "total_events"])
        w.writeheader()
        for _, row in sorted(rows.items(), key=lambda x: x[0]):
            w.writerow(row)

# =========================
# Event Manager
# =========================
class EventManager:
    def __init__(self, abnormal_s, reset_missing_s):
        self.abnormal_s = float(abnormal_s)
        self.reset_missing_s = float(reset_missing_s)
        self.state = {}  # key -> {start,last,triggered,cls}

    def touch(self, k, cls="unknown"):
        now = time.time()
        cls = str(cls)
        if k not in self.state:
            self.state[k] = {"start": now, "last": now, "triggered": False, "cls": cls}
        else:
            self.state[k]["last"] = now
            self.state[k]["cls"] = cls

    def update(self, keys_in_roi, key_to_cls):
        now = time.time()
        events = []

        for k in keys_in_roi:
            cls = str(key_to_cls.get(k, "unknown"))
            self.touch(k, cls)

            elapsed = now - self.state[k]["start"]
            if (not self.state[k]["triggered"]) and elapsed >= self.abnormal_s:
                self.state[k]["triggered"] = True
                events.append({"key": k, "cls": cls, "elapsed": elapsed})

        # reset missing
        for k in list(self.state.keys()):
            if k not in keys_in_roi:
                if (now - self.state[k]["last"]) > self.reset_missing_s:
                    del self.state[k]

        return events

    def elapsed(self, k):
        if k not in self.state:
            return 0.0
        return time.time() - self.state[k]["start"]

    def triggered(self, k):
        return (k in self.state) and bool(self.state[k]["triggered"])

# =========================
# Snapshot + Clip
# =========================
def save_snapshot(img_dir, frame, key, cls):
    os.makedirs(img_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_key = str(key).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
    path = os.path.join(img_dir, f"{ts}_key{safe_key}_{cls}.jpg")
    cv2.imwrite(path, frame)
    return path

def write_clip(frames, fps, out_path):
    if not frames:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

    if not vw.isOpened():
        return False

    for fr in frames:
        vw.write(fr)

    vw.release()
    return True

def convert_to_mp4_h264(in_avi_path):
    """Convert AVI -> MP4(H.264). Return mp4_path if success else None."""
    if not in_avi_path or not os.path.exists(in_avi_path):
        return None

    mp4_path = os.path.splitext(in_avi_path)[0] + ".mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", in_avi_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return mp4_path
    except Exception as e:
        print("ffmpeg convert failed:", e)
        return None

def safe_write_latest_jpg(frame, path):
    """
    Write latest.jpg safely on Windows.
    - write to tmp
    - check cv2.imwrite result
    - replace if tmp exists
    - never crash main loop
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        ok = cv2.imwrite(tmp_path, frame)
        if ok and os.path.exists(tmp_path):
            os.replace(tmp_path, path)
        else:
            # fallback: try direct write (still not crash)
            cv2.imwrite(path, frame)
    except Exception as e:
        print("safe_write_latest_jpg failed:", e)

# =========================
# Mouse ROI
# =========================
roi = None
sx, sy = 0, 0

def reset_roi():
    global roi
    roi = None
    if os.path.exists(ROI_PATH):
        os.remove(ROI_PATH)
    print("♻️ ROI reset: ลากกรอบใหม่ได้เลย")


def on_mouse(event, x, y, flags, param):
    global roi, sx, sy
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        roi = normalize_roi(sx, sy, x, y)
        save_roi(ROI_PATH, roi)
        print("✅ ROI saved:", roi)

# =========================
# UI Helpers
# =========================
def draw_hud(frame, cfg, show_events, recording, post_left, recent_events_local):
    h = frame.shape[0]
    y = 28

    roi_state = "ON" if cfg["roi_enabled"] else "OFF"
    txt = f"ROI:{roi_state} | Threshold:{int(cfg['abnormal_seconds'])}s | conf:{cfg['conf_thres']:.2f} | pre:{cfg['pre_seconds']:.0f}s post:{cfg['post_seconds']:.0f}s"
    cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    hint = "Keys: 1=10s 2=30s 3=60s | T=ROI | E=Events | S=Save | R=ResetROI | ESC=Exit"
    cv2.putText(frame, hint, (10, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if recording:
        cv2.putText(frame, f"STATUS: RECORDING... {max(0, post_left)}", (10, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 3)

    if show_events:
        y0 = 110
        cv2.putText(frame, "Events (latest):", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        yy = y0 + 24
        for e in list(recent_events_local):
            cv2.putText(frame, f"- {e}", (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            yy += 22
            if yy > h - 10:
                break

# =========================
# MAIN LOOP
# =========================
def main(source=0):
    global roi

    os.makedirs(EVENT_DIR, exist_ok=True)
    os.makedirs(CLIP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ensure_csv(CSV_PATH)

    # Load config + ROI
    cfg = load_config()
    print("✅ Config loaded:", cfg)
    print("Tip: press S to save settings")

    roi = load_roi(ROI_PATH)
    if roi:
        print("✅ ROI loaded:", roi)
        print("Tip: กด R เพื่อ reset ROI แล้วลากใหม่ได้")
    else:
        print("ℹ️ ยังไม่มี ROI — ลากกรอบเพื่อบันทึก (กด R เพื่อ reset ก็ได้)")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    status["connected"] = True
    status["recording"] = False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(1.0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS from camera:", fps)
    if fps is None or fps < 1 or fps > 120:
        fps = 25.0

    # pre-buffer based on config
    pre_maxlen = max(1, int(float(cfg["pre_seconds"]) * fps))
    pre_buffer = collections.deque(maxlen=pre_maxlen)

    em = EventManager(cfg["abnormal_seconds"], RESET_MISSING_SECONDS)

    # clip recording state
    recording = False
    post_frames_left = 0
    record_frames = []
    record_meta = None

    # UI: recent events on screen
    recent_events_local = collections.deque(maxlen=5)
    show_events = True

    win = "Run V2 (Tapo-like HUD)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame drop, retry...")
            time.sleep(0.05)
            continue

        # resize
        infer_frame = cv2.resize(frame, (INFER_W, INFER_H))
        disp_frame  = cv2.resize(frame, (DISP_W,  DISP_H))

        # write latest.jpg safely
        safe_write_latest_jpg(disp_frame, LATEST_JPG_PATH)

        sx_scale = DISP_W / INFER_W
        sy_scale = DISP_H / INFER_H

        pre_buffer.append(disp_frame.copy())

        # draw ROI area for display
        if roi:
            color = (0, 255, 255) if cfg["roi_enabled"] else (120, 120, 120)
            cv2.rectangle(disp_frame, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), color, 2)
            label = "WARNING ROI" if cfg["roi_enabled"] else "ROI (OFF)"
            cv2.putText(disp_frame, f"{label} (press R to reset)", (roi["x1"], max(20, roi["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            cv2.putText(disp_frame, "Draw ROI (mouse drag). Press R to reset.", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # detect+track
        results = model.track(infer_frame, persist=True, conf=float(cfg["conf_thres"]), verbose=False)

        keys_in_roi = set()
        key_to_cls = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id  # can be None
            clss = boxes.cls
            confs = boxes.conf
            xyxys = boxes.xyxy

            for i in range(len(xyxys)):
                cls_id = int(clss[i].item())
                if cls_id not in TARGET_CLASS_IDS:
                    continue

                conf = float(confs[i].item())
                x1, y1, x2, y2 = map(int, xyxys[i].tolist())

                # track id
                tid = int(ids[i].item()) if ids is not None else None
                key = tid if tid is not None else (cls_id, x1, y1, x2, y2)

                cls_name = CLS_NAME.get(cls_id, str(cls_id))
                key_to_cls[key] = cls_name

                # map bbox to display coords
                dx1 = int(x1 * sx_scale); dy1 = int(y1 * sy_scale)
                dx2 = int(x2 * sx_scale); dy2 = int(y2 * sy_scale)
                dcx, dcy = center_of_bbox(dx1, dy1, dx2, dy2)

                # draw bbox
                label = f"ID:{tid} {cls_name} {conf:.2f}" if tid is not None else f"{cls_name} {conf:.2f}"
                cv2.rectangle(disp_frame, (dx1, dy1), (dx2, dy2), (255, 255, 255), 2)
                cv2.circle(disp_frame, (dcx, dcy), 5, (255, 255, 255), -1)
                cv2.putText(disp_frame, label, (dx1, max(20, dy1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # ROI logic
                if cfg["roi_enabled"] and roi and in_roi(dcx, dcy, roi):
                    keys_in_roi.add(key)
                    em.touch(key, cls_name)
                    elapsed = em.elapsed(key)

                    cv2.putText(disp_frame, f"{elapsed:.1f}s", (dx1, dy2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if em.triggered(key):
                        cv2.putText(disp_frame, "ABNORMAL", (dx1, dy2 + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

        # ===== IMPORTANT: update events must be OUTSIDE detection loop =====
        events = em.update(keys_in_roi, key_to_cls)

        for ev in events:
            if not recording:
                snap_path = save_snapshot(EVENT_DIR, disp_frame, ev["key"], ev["cls"])
                print("✅ Snapshot saved:", snap_path)

                recording = True
                status["recording"] = True

                post_frames_left = max(1, int(float(cfg["post_seconds"]) * fps))
                record_frames = list(pre_buffer)
                record_meta = {
                    "key": ev["key"],
                    "cls": ev["cls"],
                    "elapsed": ev["elapsed"],
                    "snapshot_path": snap_path,
                }
                print(f"🚨 TRIGGER: key={ev['key']} cls={ev['cls']} | start recording clip...")

        # record post-trigger
        if recording:
            record_frames.append(disp_frame.copy())
            post_frames_left -= 1

            if post_frames_left <= 0 and record_meta is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                safe_key = (
                    str(record_meta["key"])
                    .replace(" ", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "_")
                )

                # 1) save avi
                out_avi = os.path.join(CLIP_DIR, f"{ts}_key{safe_key}_{record_meta['cls']}.avi")
                ok = write_clip(record_frames, fps, out_avi)
                print("✅ Clip saved:" if ok else "❌ Clip save failed:", out_avi)

                # 2) convert to mp4
                mp4_path = None
                if ok:
                    mp4_path = convert_to_mp4_h264(out_avi)

                final_clip_path = mp4_path if mp4_path else out_avi

                # optional: remove avi if mp4 created
                if mp4_path and os.path.exists(out_avi):
                    try:
                        os.remove(out_avi)
                    except Exception as e:
                        print("⚠️ Failed to remove original AVI:", e)

                # 3) log
                if ok:
                    append_event_csv(
                        CSV_PATH,
                        ROI_ID,
                        record_meta["key"],
                        record_meta["cls"],
                        record_meta["elapsed"],
                        record_meta["snapshot_path"],
                        final_clip_path,
                        source=str(source),
                    )
                    update_summary(SUMMARY_PATH, record_meta["cls"])

                    ev_text = f"{time.strftime('%H:%M:%S')} | {record_meta['cls']} | {record_meta['elapsed']:.0f}s"
                    status["last_event"] = ev_text
                    recent_events_local.appendleft(ev_text)

                    print("📝 Logged:", CSV_PATH)
                    print("📊 Summary:", SUMMARY_PATH)

                # reset state
                recording = False
                status["recording"] = False
                record_frames = []
                record_meta = None
                post_frames_left = 0

        # HUD
        draw_hud(disp_frame, cfg, show_events, recording, post_frames_left, recent_events_local)

        cv2.imshow(win, disp_frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k in (ord('r'), ord('R')):
            reset_roi()
        elif k in (ord('s'), ord('S')):
            save_config(cfg)
            print("💾 settings saved:", CONFIG_PATH)
        elif k in (ord('e'), ord('E')):
            show_events = not show_events
            print("🧾 show_events =", show_events)
        elif k in (ord('t'), ord('T')):
            cfg["roi_enabled"] = not bool(cfg["roi_enabled"])
            print("⚙️ roi_enabled =", cfg["roi_enabled"])
        elif k == ord('1'):
            cfg["abnormal_seconds"] = 10.0
            em.abnormal_s = 10.0
            print("⚙️ abnormal_seconds = 10 (applied)")
        elif k == ord('2'):
            cfg["abnormal_seconds"] = 30.0
            em.abnormal_s = 30.0
            print("⚙️ abnormal_seconds = 30 (applied)")
        elif k == ord('3'):
            cfg["abnormal_seconds"] = 60.0
            em.abnormal_s = 60.0
            print("⚙️ abnormal_seconds = 60 (applied)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== START RTSP MODE ===")
    main(url)
    status["connected"] = False
    status["recording"] = False