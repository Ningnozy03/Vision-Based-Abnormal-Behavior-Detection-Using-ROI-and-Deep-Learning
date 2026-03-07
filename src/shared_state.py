# shared_state.py
from collections import deque
import threading
import threading
latest_jpg = None
latest_jpg_lock = threading.Lock()

status = {
    "connected": False,
    "recording": False,
    "last_event": None,
}

recent_events = deque(maxlen=20)

# ✅ เฟรมล่าสุด (jpg bytes) ให้ API เอาไป stream
latest_jpg = None
latest_jpg_lock = threading.Lock()
