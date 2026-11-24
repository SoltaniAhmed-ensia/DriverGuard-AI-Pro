"""Create a PDF trip report for a session (charts + events + evaluation).

    python report.py                       # latest session
    python report.py sessions/20261112_101500
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from evaluate import evaluate_session, latest_session

CATS = ["drowsy", "yawning", "distracted", "phone", "absent"]
COLORS = {"drowsy": "#d62728", "yawning": "#ff7f0e", "distracted": "#1f77b4",
          "phone": "#9467bd", "absent": "#7f7f7f"}


def risk_rating(avg_attention, level2_events, max_fatigue):
    score = (100 - avg_attention) + 3 * level2_events + 0.2 * max_fatigue
    if score < 20:
        return "LOW", "#2ca02c"
    if score < 45:
        return "MEDIUM", "#ff7f0e"
    return "HIGH", "#d62728"


def build_report(d):
    d = Path(d)
    tl = pd.read_csv(d / "timeline.csv")
    ev = pd.read_csv(d / "events.csv")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) \
        if (d / "meta.json").exists() else {}
    out = d / "report.pdf"
    avg = tl["attention"].mean()
    lvl2 = int((ev["level"] >= 2).sum()) if len(ev) else 0
    risk, rcol = risk_rating(avg, lvl2, tl["fatigue"].max())

    with PdfPages(out) as pdf:
        # ---------------- page 1: summary ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.95, "DriverGuard AI Pro - Trip Report", ha="center",
                 fontsize=20, weight="bold")
        fig.text(0.5, 0.925, f"Session {d.name}   |   duration {meta.get('duration_s', tl['t'].max()):.0f} s"
                 f"   |   source: {meta.get('source', 'camera')}", ha="center", fontsize=9, color="gray")
        if meta.get("synthetic"):
            fig.text(0.5, 0.905, "SYNTHETIC SAMPLE DATA - for software testing only",
                     ha="center", fontsize=10, color="red", weight="bold")
        kp = [("Average attention", f"{avg:.0f} %"),
              ("Lowest attention", f"{tl['attention'].min():.0f} %"),
              ("Peak fatigue index", f"{tl['fatigue'].max():.0f} %"),
              ("Alarm-level events", f"{lvl2}"),
              ("Total alerts", f"{len(ev)}"),
              ("Driving risk", risk)]
        for i, (k, v) in enumerate(kp):
            x = 0.08 + (i % 3) * 0.3
            y = 0.84 - (i // 3) * 0.08
            fig.text(x, y + 0.03, k, fontsize=9, color="gray")
            fig.text(x, y, v, fontsize=18, weight="bold",
                     color=rcol if k == "Driving risk" else "black")
        ax = fig.add_axes([0.1, 0.45, 0.85, 0.22])
        ax.plot(tl["t"], tl["attention"], color="#1f77b4", label="Attention")
        ax.plot(tl["t"], tl["fatigue"], color="#d62728", label="Fatigue index")
        ax.set_ylim(0, 105)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("%")
        ax.set_title("Attention and fatigue over time")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        ax2 = fig.add_axes([0.1, 0.12, 0.85, 0.24])
        for i, c in enumerate(CATS):
            col = f"a_{c}"
            ax2.fill_between(tl["t"], i, i + tl[col] * 0.8, step="post",
                             color=COLORS[c], alpha=0.85)
        ax2.set_yticks([i + 0.4 for i in range(len(CATS))])
        ax2.set_yticklabels(CATS)
        ax2.set_xlabel("time (s)")
        ax2.set_title("Alert timeline")
        ax2.grid(alpha=0.3, axis="x")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- page 2: signals + events ----------------
        fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), sharex=True)
        fig.suptitle("Driver signals", fontsize=16, weight="bold")
        axes[0].plot(tl["t"], tl["ear"], color="#2ca02c")
        axes[0].set_ylabel("EAR (eye openness)")
        axes[1].plot(tl["t"], tl["mar"], color="#ff7f0e")
        axes[1].set_ylabel("MAR (mouth openness)")
        axes[2].plot(tl["t"], tl["yaw"], label="yaw", color="#1f77b4")
        axes[2].plot(tl["t"], tl["pitch"], label="pitch", color="#9467bd")
        axes[2].legend(loc="upper right")
        axes[2].set_ylabel("head angle (deg)")
        axes[3].plot(tl["t"], tl["perclos"] * 100, color="#d62728")
        axes[3].set_ylabel("PERCLOS (%)")
        axes[3].set_xlabel("time (s)")
        for a in axes:
            a.grid(alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.95, "Alert log", ha="center", fontsize=16, weight="bold")
        ax = fig.add_axes([0.05, 0.55, 0.9, 0.37])
        ax.axis("off")
        if len(ev):
            show = ev.head(25)
            tab = ax.table(cellText=show.values.tolist(), colLabels=list(show.columns),
                           loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            tab.scale(1, 1.25)
        else:
            ax.text(0.5, 0.5, "No alerts during this session", ha="center")
        res = evaluate_session(d)
        fig.text(0.05, 0.48, "Detection performance (from labelled test)", fontsize=13,
                 weight="bold")
        ax3 = fig.add_axes([0.05, 0.25, 0.9, 0.2])
        ax3.axis("off")
        if res["overall"]["has_labels"]:
            rows = []
            for c, r in res["per_class"].items():
                if r["labelled_events"] or r["system_alerts"]:
                    f = lambda v: "-" if v is None else f"{v * 100:.0f}%"
                    rows.append([c, r["labelled_events"], r["detected_events"], r["false_alarms"],
                                 f(r["recall"]), f(r["precision"]), f(r["f1"])])
            tab = ax3.table(cellText=rows, colLabels=["behaviour", "labelled", "detected",
                            "false alarms", "recall", "precision", "F1"], loc="upper center",
                            cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(9)
            tab.scale(1, 1.4)
        else:
            ax3.text(0.5, 0.6, "No ground-truth labels recorded.\nPress 1/2/3/4 during a test "
                     "session to label behaviours.", ha="center", color="gray")
        pdf.savefig(fig)
        plt.close(fig)
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_session()
    if d is None:
        print("No session found.")
        raise SystemExit(1)
    print("Report saved:", build_report(d))
