# DriverGuard AI Pro  -  Real-time Driver Monitoring System

Camera-based AI that watches the driver and sounds graded alarms when they are
drowsy, distracted, on the phone, or unresponsive. Everything runs **offline on a
normal laptop CPU** (no cloud, no GPU).

## 1. Install (once)

```
pip install -r requirements.txt
pip install ultralytics pyttsx3        # optional: phone detection + spoken warnings
```
Python 3.10 / 3.11 recommended. The first run with `--yolo` downloads a 6 MB model (needs internet once).

## 2. Run

| Command | What it does |
|---|---|
| `python main.py` | basic monitor (English) |
| `python main.py --lang ar --yolo --voice` | Arabic interface + phone detection + spoken warnings |
| `python main.py --lang fr` | French interface |
| `python main.py --video clip.mp4` | analyse a recorded video (backup if the camera fails) |
| `python main.py --record` | also saves an annotated video of the session |
| `streamlit run dashboard.py` | live dashboard (open next to the camera window) |
| `python report.py` | PDF trip report of the last session |
| `python evaluate.py` | precision / recall / F1 of the last labelled session |
| `python -m unittest tests.test_engine -v` | 13 automatic tests of the decision logic |

Keys in the camera window: `q` quit - `r` recalibrate - `m` mute -
`1` drowsy - `2` yawning - `3` distracted - `4` phone (ground-truth labels, see section 5).

Windows shortcut: double-click `start_windows.bat`.

## 3. What it detects

| Feature | How |
|---|---|
| Personal **calibration** (4 s) | learns *your* open-eye EAR, head pose and gaze; works with glasses / small eyes |
| **Drowsiness** | EAR (eye aspect ratio) + **PERCLOS** (standard fatigue metric) + **micro-sleeps** |
| **Blink analytics** | blink rate and blink duration per minute |
| **Fatigue Index 0-100** | fuses PERCLOS, yawns, blink duration, micro-sleeps |
| **Yawning** | MAR (mouth aspect ratio); 3 yawns/min = warning |
| **Head distraction** | 3-D head pose with solvePnP (yaw / pitch) |
| **Eyes-off-road (iris gaze)** | iris tracking: detects looking away even when the head stays straight |
| **Hand near face** | MediaPipe Hands: phone call / eating / smoking |
| **Phone use** | YOLOv8 object detector (optional `--yolo`) |
| **Driver not visible** | face lost > 2 s |
| **Night mode** | automatic low-light enhancement (CLAHE + gamma) |
| **Graded alarms** | level 1 beep -> level 2 siren -> level 3 emergency siren |
| **Emergency escalation** | alarm > 8 s with no reaction = "driver unresponsive" (hazard lights + emergency call are a *simulation* in this prototype) |
| **Voice warnings** | offline text-to-speech (optional) |
| **Black-box evidence** | automatic snapshots on every alarm |
| **Session recorder** | `sessions/<date>/` with timeline.csv, events.csv, snapshots, meta.json |
| **Live dashboard** | Streamlit, auto-refresh |
| **PDF report** | charts + alert log + risk rating |
| **Scientific evaluation** | labelled test -> precision / recall / F1 / delay / false alarms per hour |
| **3 languages** | English, Arabic (right-to-left, shaped), French |

## 4. Honest limits (say this to the jury - it earns respect)

* No AI system has zero error. Accuracy depends on light, camera angle, glasses, face shape.
  The design reduces false alarms with time windows (a glance in the mirror does not alarm), personal
  calibration, and smoothing.
* Sunglasses block eye tracking; the system then relies on head pose / presence.
* Tested in a lab (laptop webcam), not yet in a moving vehicle.
* Emergency actions (hazard lights, SMS) are simulated.

## 5. How to measure your accuracy (do this before the defense!)

1. Run `python main.py --lang en` and press `r` to calibrate.
2. Act each behaviour **on purpose, 8-10 times**: press the label key when you start
   (`1` drowsy / `2` yawning / `3` looking away / `4` phone) and press it again when you stop.
3. Quit with `q`, then run `python evaluate.py` -> put the table in your slides.
4. Repeat in a dark room and with glasses to show robustness.

The result is a real, defensible number such as "recall 92 %, precision 95 %, mean delay 1.4 s".

## 6. Demo script (5 minutes)

1. 30 s - the problem: fatigue/distraction cause a large share of road accidents.
2. 30 s - architecture (camera -> face mesh -> metrics -> decision engine -> alarm/log/dashboard).
3. 2 min - **live demo**: calibrate -> look aside (warning, then alarm) -> close eyes -> yawn x3 -> phone/hand
   -> leave the frame -> keep eyes closed 8 s to trigger the emergency mode.
4. 1 min - dashboard + PDF report + snapshots.
5. 1 min - results table from `evaluate.py` + limits + future work (Raspberry Pi in the car, seat-belt detection, CAN-bus integration, real SMS).

## 7. Project structure

```
main.py               entry point
driverguard/          vision.py (landmarks, EAR/MAR, pose, gaze, hands, YOLO)
                      engine.py (calibration, timers, fatigue index, alerts)
                      alerts.py (audio + voice)   hud.py (display, Arabic)
                      session.py (recording)      i18n.py (EN/AR/FR)   config.py (all thresholds)
dashboard.py          live Streamlit dashboard
report.py             PDF report              evaluate.py   precision / recall / F1
tests/                unit tests              make_sample_session.py   synthetic data for testing
```
All thresholds are in `driverguard/config.py` - tune them to your lighting if needed.

## 8. Troubleshooting

* Camera not opening -> `python main.py --camera 1`
* Arabic shows boxes -> `pip install arabic-reshaper python-bidi` (falls back to English automatically)
* `mediapipe` error -> use Python 3.10/3.11 and `pip install "mediapipe==0.10.14" "numpy<2"`
* No sound -> check volume; `python main.py` still works silently
* Too many false alarms -> increase `yaw_limit`, `gaze_x_limit` in `config.py`; recalibrate with `r`
