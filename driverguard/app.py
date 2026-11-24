"""Main application loop."""
import argparse
import collections
import time

import cv2

from . import i18n
from .alerts import Alarm, Voice
from .config import Config
from .engine import Engine
from .hud import TextLayer, draw_hud, draw_overlays
from .session import Session, GT_CATS
from .vision import FaceAnalyzer, PhoneDetector, enhance_low_light

GT_KEYS = {ord("1"): "drowsy", ord("2"): "yawning", ord("3"): "distracted", ord("4"): "phone"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DriverGuard AI Pro")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--video", type=str, default=None, help="video file instead of camera")
    ap.add_argument("--lang", choices=["en", "ar", "fr"], default="en")
    ap.add_argument("--yolo", "--phone", dest="yolo", action="store_true",
                    help="enable YOLOv8 phone detection")
    ap.add_argument("--no-hands", action="store_true", help="disable hand tracking")
    ap.add_argument("--voice", action="store_true", help="spoken warnings (pyttsx3)")
    ap.add_argument("--record", action="store_true", help="save annotated video")
    ap.add_argument("--no-window", action="store_true", help="headless (testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument("--sessions-dir", default="sessions")
    ap.add_argument("--width", type=int, default=1280)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config()
    layer = TextLayer(args.lang)
    lang = args.lang
    if not layer.available and lang != "en":
        print(f"[WARN] Cannot render '{lang}' text on this PC (font/library missing) -> English")
        lang = "en"
        layer = TextLayer("en")
    tr = i18n.get(lang)

    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
    if not cap.isOpened():
        print("Cannot open the video source. Try --camera 1 or --video file.mp4")
        return 1

    analyzer = FaceAnalyzer(use_hands=not args.no_hands)
    phone = PhoneDetector() if args.yolo else None
    engine = Engine(cfg)
    alarm = Alarm()
    voice = Voice(args.voice, tr["voice"])
    session = Session(args.sessions_dir, meta={
        "lang": lang, "yolo": args.yolo, "hands": not args.no_hands,
        "source": "video" if args.video else "camera", "config": cfg.as_dict()})

    if args.video:
        session.t0 = 0.0                          # timeline follows video time
    writer = None
    hist = collections.deque(maxlen=120)        # ~60 s at 0.5 s steps
    last_hist = 0.0
    log = []
    prev_keys = set()
    fps, last_t, n = 0.0, time.time(), 0
    snap_pending = None

    print("Keys: q quit | r recalibrate | m mute | 1 drowsy 2 yawning 3 distracted "
          "4 phone (toggle ground-truth label)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if args.video is None:
            frame = cv2.flip(frame, 1)               # mirror view for the driver
        if frame.shape[1] > args.width:
            sc = args.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=sc, fy=sc)
        now = time.time() if not args.video else n / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        wall = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(wall - last_t, 1e-3))
        last_t = wall

        frame, dark = enhance_low_light(frame, cfg.dark_brightness)
        m = analyzer.process(frame)
        boxes = []
        if phone is not None:
            m.phone, boxes = phone.update(frame, now)
        st = engine.update(now, m)

        alarm.update(st.level, wall)
        keys = {a.key for a in st.alerts}
        for a in st.alerts:
            if a.key not in prev_keys:
                session.log_event(now, a)
                log.append((session.rel(now), a.key, a.level))
                if a.level >= 2:
                    snap_pending = a.key
                    voice.say(a.key, wall)
        if "fatigue" in keys and "fatigue" not in prev_keys:
            voice.say("fatigue", wall)
        prev_keys = keys
        session.log_state(now, st, cfg.log_interval_s)

        if now - last_hist >= 0.5:
            hist.append(st.attention)
            last_hist = now

        draw_overlays(frame, m, st)
        ctx = dict(fps=fps, muted=alarm.muted, dark=dark, gt=session.gt, log=log,
                   hist=hist, now=wall, phone_boxes=boxes, calib_s=cfg.calib_seconds)
        frame = draw_hud(frame, st, m, tr, layer, ctx)

        if snap_pending:
            session.snapshot(now, frame, snap_pending)
            snap_pending = None
        if args.record:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(session.dir / "recording.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            writer.write(frame)

        if not args.no_window:
            cv2.imshow("DriverGuard AI Pro", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                alarm.muted = not alarm.muted
            if key == ord("r"):
                engine.reset_calibration()
            if key in GT_KEYS:
                session.toggle_gt(GT_KEYS[key])
        if args.max_frames and n >= args.max_frames:
            break

    alarm.stop()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    meta = session.close()

    print("\n=========== SESSION REPORT ===========")
    print(f"Duration       : {meta['duration_s']} s")
    print(f"Avg attention  : {meta['avg_attention']} %   (min {meta['min_attention']} %)")
    print(f"Max fatigue    : {meta['max_fatigue']} %")
    for k, v in meta["event_counts"].items():
        print(f"  {k:16s}: {v}")
    print(f"Saved in       : {session.dir}")
    print("Next: python report.py   |   python evaluate.py   |   streamlit run dashboard.py")
    return 0
