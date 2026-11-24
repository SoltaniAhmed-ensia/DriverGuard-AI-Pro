"""Scientific evaluation of a recorded session.

During a test session you press 1/2/3/4 to label the *true* behaviour while you
act it out (1 drowsy, 2 yawning, 3 distracted, 4 phone). This script compares
those labels with what the system detected and reports, per behaviour:
  recall (detected / acted), precision, F1, mean detection delay, false alarms/hour.

    python evaluate.py                      # latest session
    python evaluate.py sessions/20261112_101500
"""
import json
import sys
from pathlib import Path

import pandas as pd

CATS = ["drowsy", "yawning", "distracted", "phone"]
TOL = 1.5        # seconds of tolerance (you press the key slightly late/early)
MERGE_GAP = 1.0  # merge intervals separated by less than this


def intervals(t, flag):
    out, start, last = [], None, None
    for ti, f in zip(t, flag):
        if f and start is None:
            start = ti
        if f:
            last = ti
        if not f and start is not None:
            out.append([start, last])
            start = None
    if start is not None:
        out.append([start, last])
    merged = []
    for s, e in out:
        if merged and s - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def overlap(a, b, tol=TOL):
    return a[0] <= b[1] + tol and a[1] >= b[0] - tol


def evaluate_session(session_dir):
    df = pd.read_csv(Path(session_dir) / "timeline.csv")
    hours = max(df["t"].max() - df["t"].min(), 1) / 3600.0
    rows = {}
    for c in CATS:
        gt = intervals(df["t"], df[f"g_{c}"] > 0)
        al = intervals(df["t"], df[f"a_{c}"] > 0)
        hit = [g for g in gt if any(overlap(g, a) for a in al)]
        good = [a for a in al if any(overlap(a, g) for g in gt)]
        delays = []
        for g in hit:
            first = min((a[0] for a in al if overlap(g, a)), default=None)
            if first is not None:
                delays.append(max(first - g[0], 0.0))
        rec = len(hit) / len(gt) if gt else None
        pre = len(good) / len(al) if al else (1.0 if gt else None)
        f1 = (2 * pre * rec / (pre + rec)) if rec and pre else (0.0 if gt else None)
        rows[c] = {
            "labelled_events": len(gt), "detected_events": len(hit),
            "system_alerts": len(al), "false_alarms": len(al) - len(good),
            "recall": rec, "precision": pre, "f1": f1,
            "mean_delay_s": (sum(delays) / len(delays)) if delays else None,
            "false_alarms_per_hour": (len(al) - len(good)) / hours,
        }
    tot_gt = sum(r["labelled_events"] for r in rows.values())
    tot_hit = sum(r["detected_events"] for r in rows.values())
    tot_al = sum(r["system_alerts"] for r in rows.values())
    tot_fa = sum(r["false_alarms"] for r in rows.values())
    overall = {
        "recall": tot_hit / tot_gt if tot_gt else None,
        "precision": (tot_al - tot_fa) / tot_al if tot_al else (1.0 if tot_gt else None),
        "has_labels": tot_gt > 0,
    }
    if overall["recall"] and overall["precision"]:
        overall["f1"] = 2 * overall["recall"] * overall["precision"] / (
            overall["recall"] + overall["precision"])
    else:
        overall["f1"] = None
    return {"per_class": rows, "overall": overall}


def latest_session(root="sessions"):
    dirs = sorted(p for p in Path(root).glob("*") if (p / "timeline.csv").exists())
    return dirs[-1] if dirs else None


def fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x * 100:5.1f}%" if pct else f"{x:5.2f}"


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_session()
    if d is None:
        print("No session found. Run main.py first.")
        return 1
    res = evaluate_session(d)
    print(f"Session: {d}\n")
    print(f"{'behaviour':12s}{'labelled':>9s}{'detected':>9s}{'alerts':>8s}{'false':>7s}"
          f"{'recall':>9s}{'precision':>10s}{'F1':>8s}{'delay(s)':>10s}")
    for c, r in res["per_class"].items():
        print(f"{c:12s}{r['labelled_events']:9d}{r['detected_events']:9d}"
              f"{r['system_alerts']:8d}{r['false_alarms']:7d}"
              f"{fmt(r['recall'], True):>9s}{fmt(r['precision'], True):>10s}"
              f"{fmt(r['f1'], True):>8s}{fmt(r['mean_delay_s']):>10s}")
    o = res["overall"]
    if not o["has_labels"]:
        print("\nNo ground-truth labels in this session. While acting a behaviour, press "
              "1/2/3/4 to start and again to stop the label.")
    else:
        print(f"\nOVERALL  recall {fmt(o['recall'], True)}  precision "
              f"{fmt(o['precision'], True)}  F1 {fmt(o['f1'], True)}")
    (d / "evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
