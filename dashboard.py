"""Live dashboard:  streamlit run dashboard.py
Open it next to the camera window: it refreshes every 2 s while a session is running."""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluate import evaluate_session
from report import build_report, risk_rating

st.set_page_config(page_title="DriverGuard AI Pro", page_icon="🚗", layout="wide")
st.title("🚗 DriverGuard AI Pro - Dashboard")

root = Path("sessions")
sessions = sorted([p for p in root.glob("*") if (p / "timeline.csv").exists()], reverse=True)
if not sessions:
    st.info("No sessions yet. Run `python main.py` first (or `python make_sample_session.py`).")
    st.stop()

with st.sidebar:
    st.header("Session")
    choice = st.selectbox("Choose", sessions, format_func=lambda p: p.name)
    live = st.checkbox("Live refresh (2 s)", value=(choice == sessions[0]))
    st.caption("Tip: keep this open during the demo - it updates while you drive.")

d = Path(choice)
tl = pd.read_csv(d / "timeline.csv")
ev = pd.read_csv(d / "events.csv")
meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}

if meta.get("synthetic"):
    st.warning("SYNTHETIC SAMPLE DATA - generated for testing the software, not real results.")
if tl.empty:
    st.info("Waiting for data...")
else:
    last = tl.iloc[-1]
    avg = float(tl["attention"].mean())
    lvl2 = int((ev["level"] >= 2).sum()) if len(ev) else 0
    risk, _ = risk_rating(avg, lvl2, float(tl["fatigue"].max()))
    c = st.columns(6)
    c[0].metric("Attention now", f"{last['attention']:.0f}%")
    c[1].metric("Fatigue now", f"{last['fatigue']:.0f}%")
    c[2].metric("Avg attention", f"{avg:.0f}%")
    c[3].metric("Alarm events", lvl2)
    c[4].metric("Blinks / min", f"{last['blink_rate']:.0f}")
    c[5].metric("Driving risk", risk)

    left, right = st.columns(2)
    with left:
        st.subheader("Attention & fatigue")
        st.line_chart(tl.set_index("t")[["attention", "fatigue"]])
        st.subheader("Eye / mouth openness")
        st.line_chart(tl.set_index("t")[["ear", "mar"]])
    with right:
        st.subheader("Head pose (deg)")
        st.line_chart(tl.set_index("t")[["yaw", "pitch"]])
        st.subheader("Alerts over time")
        al = tl.set_index("t")[[c for c in tl.columns if c.startswith("a_")]]
        st.area_chart(al)

    st.subheader("Alert log")
    st.dataframe(ev.iloc[::-1], height=220)

    snaps = sorted((d / "snapshots").glob("*.jpg"), reverse=True)[:6]
    if snaps:
        st.subheader("Evidence snapshots (taken automatically on alarms)")
        cols = st.columns(3)
        for i, s in enumerate(snaps):
            cols[i % 3].image(str(s), caption=s.stem)

    res = evaluate_session(d)
    if res["overall"]["has_labels"]:
        st.subheader("Detection performance (from your labelled test)")
        rows = []
        for k, r in res["per_class"].items():
            rows.append({"behaviour": k, "labelled": r["labelled_events"],
                         "detected": r["detected_events"], "false alarms": r["false_alarms"],
                         "recall %": None if r["recall"] is None else round(r["recall"] * 100, 1),
                         "precision %": None if r["precision"] is None else round(r["precision"] * 100, 1),
                         "F1 %": None if r["f1"] is None else round(r["f1"] * 100, 1),
                         "mean delay (s)": None if r["mean_delay_s"] is None else round(r["mean_delay_s"], 2)})
        st.dataframe(pd.DataFrame(rows))
        o = res["overall"]
        st.success(f"Overall recall {o['recall'] * 100:.0f}% | precision "
                   f"{(o['precision'] or 0) * 100:.0f}% | F1 {(o['f1'] or 0) * 100:.0f}%")

    if (d / "meta.json").exists() and st.button("Create PDF report"):
        pdf = build_report(d)
        st.download_button("Download report.pdf", data=Path(pdf).read_bytes(),
                           file_name=f"driver_report_{d.name}.pdf", mime="application/pdf")

if live:
    time.sleep(2)
    st.rerun()
