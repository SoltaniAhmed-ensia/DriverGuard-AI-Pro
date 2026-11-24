"""Creates a SYNTHETIC session (fake data) so you can test the dashboard / report /
evaluation without a camera. It is clearly marked as synthetic everywhere.
Never present it as real results."""
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np

from driverguard.session import TIMELINE_COLS

random.seed(7)
np.random.seed(7)
d = Path("sessions") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_SYNTHETIC")
(d / "snapshots").mkdir(parents=True, exist_ok=True)
T = 180
ts = np.arange(0, T, 0.5)
# (category, start, end, alert_delay)
episodes = [("distracted", 20, 26, 1.0), ("drowsy", 45, 52, 1.4), ("yawning", 70, 82, 2.0),
            ("phone", 100, 108, 1.1), ("distracted", 130, 135, 1.0), ("drowsy", 150, 160, 1.5)]
rows, events = [], []
for t in ts:
    a = {c: 0 for c in ["drowsy", "yawning", "distracted", "phone", "absent"]}
    g = {c: 0 for c in ["drowsy", "yawning", "distracted", "phone"]}
    for c, s, e, dl in episodes:
        if s <= t <= e:
            g[c] = 1
        if s + dl <= t <= e + 0.5:
            a[c] = 1
    if 160 <= t <= 161.5:      # one deliberate false alarm
        a["distracted"] = 1
    att = 100 - 45 * (a["distracted"] + a["drowsy"] + a["phone"]) - 15 * a["yawning"]
    att = min(max(att + np.random.randn() * 2, 5), 100)
    fat = min(100, 8 + 0.25 * t * (1 + a["drowsy"]) / 2 + np.random.randn())
    rows.append([t, att, fat, 0.30 - 0.2 * a["drowsy"] + np.random.randn() * .01,
                 0.05 + 0.7 * a["yawning"], 35 * a["distracted"], 3, 0.0, 0.1 * a["drowsy"], 15,
                 2 if (a["drowsy"] or a["phone"]) else int(any(a.values())),
                 *a.values(), *g.values()])
import csv
with open(d / "timeline.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(TIMELINE_COLS)
    w.writerows([[round(x, 3) if isinstance(x, float) else x for x in r] for r in rows])
with open(d / "events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "key", "level", "category"])
    keymap = {"distracted": "look_road", "drowsy": "eyes_closed", "yawning": "yawning", "phone": "phone"}
    for c, s, e, dl in episodes:
        w.writerow([s + dl, keymap[c], 1 if c == "yawning" else 2, c])
    w.writerow([160, "look_road_warn", 1, "distracted"])
(d / "meta.json").write_text(json.dumps({
    "synthetic": True, "duration_s": T, "avg_attention": 78.0, "min_attention": 5.0,
    "max_fatigue": 60.0, "source": "synthetic", "lang": "en"}, indent=2))
print("Created", d)
