"""Session recorder: events.csv, timeline.csv, snapshots/, meta.json."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from .i18n import CATS

GT_CATS = ["drowsy", "yawning", "distracted", "phone"]
TIMELINE_COLS = (["t", "attention", "fatigue", "ear", "mar", "yaw", "pitch", "gaze_x",
                  "perclos", "blink_rate", "level"] +
                 [f"a_{c}" for c in CATS] + [f"g_{c}" for c in GT_CATS])


class Session:
    def __init__(self, root="sessions", meta=None):
        self.dir = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S")
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.meta = dict(meta or {})
        self.meta["started"] = datetime.now().isoformat(timespec="seconds")
        self._ev = open(self.dir / "events.csv", "w", newline="", encoding="utf-8")
        self._tl = open(self.dir / "timeline.csv", "w", newline="", encoding="utf-8")
        self.ev = csv.writer(self._ev)
        self.tl = csv.writer(self._tl)
        self.ev.writerow(["t", "key", "level", "category"])
        self.tl.writerow(TIMELINE_COLS)
        self.gt = {c: False for c in GT_CATS}
        self.last_log = -1e9
        self.last_snap = -1e9
        self.n_snap = 0
        self.attn_sum = 0.0
        self.attn_n = 0
        self.max_fatigue = 0.0
        self.min_attn = 100.0
        self.counts = {}
        self.last_rel = 0.0

    def rel(self, now):
        return now - self.t0

    def toggle_gt(self, cat):
        self.gt[cat] = not self.gt[cat]
        return self.gt[cat]

    def log_event(self, now, alert):
        self.ev.writerow([round(self.rel(now), 2), alert.key, alert.level, alert.cat])
        self._ev.flush()
        self.counts[alert.key] = self.counts.get(alert.key, 0) + 1

    def log_state(self, now, st, interval):
        self.last_rel = self.rel(now)
        if st.calibrated:
            self.attn_sum += st.attention
            self.attn_n += 1
            self.max_fatigue = max(self.max_fatigue, st.fatigue)
            self.min_attn = min(self.min_attn, st.attention)
        if now - self.last_log < interval:
            return
        self.last_log = now
        cats = {a.cat for a in st.alerts}
        row = [round(self.rel(now), 2), round(st.attention, 1), round(st.fatigue, 1),
               round(st.ear, 3), round(st.mar, 3), round(st.dyaw, 1), round(st.dpitch, 1),
               round(st.dgx, 3), round(st.perclos, 3), round(st.blink_rate, 1), st.level]
        row += [int(c in cats) for c in CATS]
        row += [int(self.gt[c]) for c in GT_CATS]
        self.tl.writerow(row)
        self._tl.flush()

    def snapshot(self, now, frame, key):
        if now - self.last_snap < 4.0 or self.n_snap >= 60:
            return
        self.last_snap = now
        self.n_snap += 1
        name = f"{self.rel(now):07.1f}s_{key}.jpg"
        cv2.imwrite(str(self.dir / "snapshots" / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

    def close(self):
        dur = self.last_rel
        self.meta.update({
            "duration_s": round(dur, 1),
            "avg_attention": round(self.attn_sum / max(self.attn_n, 1), 1),
            "min_attention": round(self.min_attn, 1),
            "max_fatigue": round(self.max_fatigue, 1),
            "event_counts": self.counts,
        })
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2),
                                            encoding="utf-8")
        self._ev.close()
        self._tl.close()
        return self.meta
