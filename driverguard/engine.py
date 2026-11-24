"""Decision engine: calibration, timers, blink analytics, fatigue index,
graded alerts, attention score and emergency escalation."""
import collections
from dataclasses import dataclass, field

import numpy as np

from .config import Config


class Timer:
    """How long a condition has been continuously true."""

    def __init__(self):
        self.start = None

    def update(self, cond, now):
        if cond:
            if self.start is None:
                self.start = now
            return now - self.start
        self.start = None
        return 0.0


@dataclass
class Alert:
    key: str       # message key (i18n)
    level: int     # 1 warning, 2 alarm, 3 emergency
    cat: str       # drowsy | yawning | distracted | phone | absent | emergency


@dataclass
class State:
    calibrated: bool = False
    calib_progress: float = 0.0
    face: bool = False
    alerts: list = field(default_factory=list)
    level: int = 0
    ear: float = 0.0
    ear_thr: float = 0.0
    mar: float = 0.0
    dyaw: float = 0.0
    dpitch: float = 0.0
    dgx: float = 0.0
    dgy: float = 0.0
    closed_d: float = 0.0
    away_d: float = 0.0
    gaze_d: float = 0.0
    hand_d: float = 0.0
    phone_d: float = 0.0
    noface_d: float = 0.0
    perclos: float = 0.0
    yawns: int = 0
    blink_rate: float = 0.0
    blink_dur: float = 0.0
    micro: int = 0
    fatigue: float = 0.0
    attention: float = 100.0


def _clip01(x):
    return max(0.0, min(1.0, x))


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset_calibration()
        self.t_closed, self.t_away, self.t_gaze = Timer(), Timer(), Timer()
        self.t_hand, self.t_phone, self.t_noface = Timer(), Timer(), Timer()
        self.t_yawn, self.t_emerg = Timer(), Timer()
        self.perclos_buf = collections.deque()
        self.yawns = collections.deque()
        self.blinks = collections.deque()      # (t, duration)
        self.micros = collections.deque()      # (t, duration)
        self.ear_hist = collections.deque(maxlen=3)
        self.yawn_counted = False
        self.close_start = None
        self.s_pitch = self.s_yaw = self.s_gx = self.s_gy = 0.0
        self.smooth_init = False
        self.fatigue = 0.0
        self.attention = 100.0
        self.last_t = None
        self.done_at = None
        self.ear = 0.0

    # ------------------------------------------------------------ calibration
    def reset_calibration(self):
        self.calib = {"start": None, "ear": [], "pitch": [], "yaw": [], "gx": [], "gy": []}
        self.done = False
        self.base_ear, self.base_pitch, self.base_yaw = 0.28, 0.0, 0.0
        self.base_gx, self.base_gy = 0.0, 0.0
        self.ear_thr = 0.20

    def _calibrate(self, now, m):
        c = self.calib
        if c["start"] is None:
            c["start"] = now
        if m.ear > 0.15:                      # ignore blinks while calibrating
            c["ear"].append(self.ear)
        c["pitch"].append(self.s_pitch)
        c["yaw"].append(self.s_yaw)
        c["gx"].append(self.s_gx)
        c["gy"].append(self.s_gy)
        if now - c["start"] >= self.cfg.calib_seconds:
            if len(c["ear"]) < 20 or np.median(c["ear"]) < 0.15:
                self.reset_calibration()      # eyes were closed / bad data -> retry
                return
            self.base_ear = float(np.median(c["ear"]))
            self.base_pitch = float(np.median(c["pitch"]))
            self.base_yaw = float(np.median(c["yaw"]))
            self.base_gx = float(np.median(c["gx"]))
            self.base_gy = float(np.median(c["gy"]))
            self.ear_thr = min(max(self.base_ear * self.cfg.ear_ratio, 0.12), 0.25)
            self.done = True
            self.done_at = now

    # ----------------------------------------------------------------- update
    def update(self, now, m):
        c = self.cfg
        dt = 0.033 if self.last_t is None else max(now - self.last_t, 1e-3)
        self.last_t = now
        st = State(calibrated=self.done, face=m.face)

        if m.face:
            if not self.smooth_init:
                self.s_pitch, self.s_yaw = m.pitch, m.yaw
                self.s_gx, self.s_gy = m.gaze_x, m.gaze_y
                self.smooth_init = True
            self.ear_hist.append(m.ear)
            self.ear = float(np.mean(self.ear_hist))
            self.s_pitch = 0.6 * self.s_pitch + 0.4 * m.pitch
            self.s_yaw = 0.6 * self.s_yaw + 0.4 * m.yaw
            self.s_gx = 0.6 * self.s_gx + 0.4 * m.gaze_x
            self.s_gy = 0.6 * self.s_gy + 0.4 * m.gaze_y
            if not self.done:
                self._calibrate(now, m)
                st.calib_progress = _clip01(
                    (now - (self.calib["start"] or now)) / c.calib_seconds)
        else:
            self.ear_hist.clear()
            if not self.done:
                self.calib["start"] = None

        det = self.done and m.face
        st.calibrated = self.done
        st.ear, st.ear_thr, st.mar = (self.ear if m.face else 0.0), self.ear_thr, m.mar
        st.dyaw = self.s_yaw - self.base_yaw if det else 0.0
        st.dpitch = self.s_pitch - self.base_pitch if det else 0.0
        st.dgx = self.s_gx - self.base_gx if det else 0.0
        st.dgy = self.s_gy - self.base_gy if det else 0.0

        eyes_closed = det and self.ear < self.ear_thr

        # ---- blink / micro-sleep analytics ----
        if det and eyes_closed:
            if self.close_start is None:
                self.close_start = now
        else:
            if self.close_start is not None and det:
                d = now - self.close_start
                if d >= 0.06:
                    (self.blinks if d < c.microsleep_s else self.micros).append((now, d))
            self.close_start = None
        while self.blinks and now - self.blinks[0][0] > 60:
            self.blinks.popleft()
        while self.micros and now - self.micros[0][0] > 120:
            self.micros.popleft()
        elapsed = (now - self.done_at) if self.done_at else 0
        window = max(min(elapsed, 60.0), 10.0)
        st.blink_rate = len(self.blinks) * 60.0 / window if self.done else 0.0
        st.blink_dur = float(np.mean([d for _, d in self.blinks])) if self.blinks else 0.0
        st.micro = len(self.micros)

        # ---- timers ----
        st.closed_d = self.t_closed.update(eyes_closed, now)
        head_away = det and (abs(st.dyaw) > c.yaw_limit or abs(st.dpitch) > c.pitch_limit)
        st.away_d = self.t_away.update(head_away, now)
        gaze_away = (det and m.gaze_ok and not eyes_closed and not head_away and
                     (abs(st.dgx) > c.gaze_x_limit or abs(st.dgy) > c.gaze_y_limit))
        st.gaze_d = self.t_gaze.update(gaze_away, now)
        st.hand_d = self.t_hand.update(det and m.hand_near_face, now)
        st.phone_d = self.t_phone.update(self.done and m.phone, now)
        st.noface_d = self.t_noface.update(self.done and not m.face, now)

        yawn_d = self.t_yawn.update(det and m.mar > c.mar_yawn, now)
        if yawn_d >= c.yawn_min_s and not self.yawn_counted:
            self.yawns.append(now)
            self.yawn_counted = True
        if yawn_d == 0:
            self.yawn_counted = False
        while self.yawns and now - self.yawns[0] > 60:
            self.yawns.popleft()
        st.yawns = len(self.yawns)

        # ---- PERCLOS ----
        if det:
            self.perclos_buf.append((now, eyes_closed))
        while self.perclos_buf and now - self.perclos_buf[0][0] > c.perclos_window_s:
            self.perclos_buf.popleft()
        if len(self.perclos_buf) > 30:
            span = self.perclos_buf[-1][0] - self.perclos_buf[0][0]
            if span > 10:
                st.perclos = sum(x for _, x in self.perclos_buf) / len(self.perclos_buf)

        # ---- fatigue index (0-100) ----
        if self.done:
            target = (40 * _clip01(st.perclos / c.perclos_limit) +
                      20 * _clip01(st.yawns / c.yawns_limit) +
                      20 * _clip01((st.blink_dur - 0.15) / 0.25) +
                      20 * _clip01(st.micro / 2.0))
            self.fatigue += (target - self.fatigue) * min(dt * 0.5, 1.0)
        st.fatigue = self.fatigue

        # ---- alerts ----
        A = st.alerts
        if self.done:
            if st.closed_d >= c.closed_alarm_s:
                A.append(Alert("eyes_closed", 2, "drowsy"))
            elif st.perclos > c.perclos_limit:
                A.append(Alert("perclos", 1, "drowsy"))
            if self.fatigue >= c.fatigue_warn and not any(a.cat == "drowsy" for a in A):
                A.append(Alert("fatigue", 1, "drowsy"))
            if st.yawns >= c.yawns_limit:
                A.append(Alert("yawning", 1, "yawning"))
            if st.away_d >= c.away_alarm_s:
                A.append(Alert("look_road", 2, "distracted"))
            elif st.away_d >= c.away_warn_s:
                A.append(Alert("look_road_warn", 1, "distracted"))
            if st.gaze_d >= c.gaze_alarm_s:
                A.append(Alert("gaze_away", 2, "distracted"))
            elif st.gaze_d >= c.gaze_warn_s:
                A.append(Alert("gaze_warn", 1, "distracted"))
            if st.phone_d >= c.phone_alarm_s:
                A.append(Alert("phone", 2, "phone"))
            elif st.hand_d >= c.hand_face_s:
                A.append(Alert("hand_face", 1, "phone"))
            if st.noface_d >= c.noface_alarm_s:
                A.append(Alert("no_face", 2, "absent"))

        lvl = max([a.level for a in A], default=0)
        emerg_d = self.t_emerg.update(lvl >= 2, now)
        if emerg_d >= c.emergency_s:
            A.insert(0, Alert("emergency", 3, "emergency"))
            lvl = 3
        st.level = lvl

        # ---- attention score ----
        if self.done:
            t = 100.0
            t -= 35 * _clip01(st.closed_d / c.closed_alarm_s)
            t -= 30 * _clip01(st.away_d / c.away_alarm_s)
            t -= 20 * _clip01(st.gaze_d / c.gaze_alarm_s)
            t -= 15 * _clip01(st.hand_d / c.hand_face_s)
            t -= 40 if st.phone_d > 0 else 0
            t -= 50 if st.noface_d > 0 else 0
            t -= 0.3 * self.fatigue
            self.attention += (max(t, 0.0) - self.attention) * min(dt * 3.0, 1.0)
        st.attention = self.attention
        return st
