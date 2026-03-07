import time

class EventManager:
    def __init__(self, abnormal_seconds=30.0, reset_missing_seconds=2.0):
        self.abnormal_seconds = float(abnormal_seconds)
        self.reset_missing_seconds = float(reset_missing_seconds)
        # key -> {start_time, last_seen, triggered, cls}
        self.state = {}

    def touch(self, k, cls="unknown"):
        """
        เรียกทันทีเมื่อ key อยู่ใน ROI (ในเฟรมนี้)
        เพื่อให้ get_elapsed(k) เดินได้ทันที ไม่ต้องรอ update() รอบถัดไป
        """
        now = time.time()
        cls = str(cls)

        if k not in self.state:
            self.state[k] = {
                "start_time": now,
                "last_seen": now,
                "triggered": False,
                "cls": cls,
            }
        else:
            self.state[k]["last_seen"] = now
            self.state[k]["cls"] = cls

    def update(self, keys_in_roi, key_to_cls: dict):
        """
        keys_in_roi: set/list ของ key ที่อยู่ใน ROI ณ เฟรมนี้
        key_to_cls: dict key -> cls_name/cls_id
        return: events = [{"key", "cls", "elapsed"}]
        """
        now = time.time()
        events = []

        # 1) update seen + trigger check
        for k in keys_in_roi:
            cls = str(key_to_cls.get(k, "unknown"))
            # สร้าง/อัปเดต state ให้แน่นอน
            self.touch(k, cls)

            elapsed = now - self.state[k]["start_time"]
            if (not self.state[k]["triggered"]) and elapsed >= self.abnormal_seconds:
                self.state[k]["triggered"] = True
                events.append({"key": k, "cls": cls, "elapsed": elapsed})

        # 2) reset missing (ออก ROI นานเกิน reset_missing_seconds)
        # ใช้ list(...) เพื่อไม่แก้ dict ตอนวน
        for k in list(self.state.keys()):
            if k not in keys_in_roi:
                last_seen = self.state[k]["last_seen"]
                if (now - last_seen) > self.reset_missing_seconds:
                    del self.state[k]

        return events

    def get_elapsed(self, k):
        """
        เวลาอยู่ใน ROI ต่อเนื่องของ key นี้ (วินาที)
        """
        if k not in self.state:
            return 0.0
        return time.time() - self.state[k]["start_time"]

    def is_triggered(self, k):
        return (k in self.state) and bool(self.state[k]["triggered"])

    def reset(self, k):
        """
        ถ้าต้องการล้าง state ของ key แบบ manual
        """
        if k in self.state:
            del self.state[k]
