"""Unit tests for the decision engine (no camera needed).
Run:  python -m unittest tests.test_engine -v
"""
import unittest

from driverguard.config import Config
from driverguard.engine import Engine
from driverguard.vision import Metrics

FPS = 30


def frame(**kw):
    base = dict(face=True, ear=0.30, mar=0.05, pitch=0.0, yaw=0.0,
                gaze_x=0.0, gaze_y=0.0, gaze_ok=True)
    base.update(kw)
    return Metrics(**base)


def run(engine, t0, seconds, **kw):
    """Feed `seconds` of identical frames; return (last_state, end_time, all_keys)."""
    keys, st, t = set(), None, t0
    for _ in range(int(seconds * FPS)):
        t += 1.0 / FPS
        st = engine.update(t, frame(**kw))
        keys |= {a.key for a in st.alerts}
    return st, t, keys


def calibrated_engine():
    e = Engine(Config())
    _, t, _ = run(e, 0.0, 5.0)
    assert e.done
    return e, t


class EngineTests(unittest.TestCase):
    def test_calibration_sets_personal_threshold(self):
        e, _ = calibrated_engine()
        self.assertAlmostEqual(e.base_ear, 0.30, places=2)
        self.assertAlmostEqual(e.ear_thr, 0.30 * 0.72, places=2)

    def test_normal_driving_gives_no_alert(self):
        e, t = calibrated_engine()
        st, t, keys = run(e, t, 20.0)
        self.assertEqual(keys, set())
        self.assertGreater(st.attention, 95)

    def test_normal_blinks_do_not_alarm(self):
        e, t = calibrated_engine()
        allk = set()
        for _ in range(10):
            _, t, k = run(e, t, 0.2, ear=0.05)     # 0.2 s blink
            allk |= k
            _, t, k = run(e, t, 2.0)
            allk |= k
        self.assertEqual(allk, set())

    def test_eyes_closed_alarm(self):
        e, t = calibrated_engine()
        st, t, keys = run(e, t, 2.0, ear=0.05)
        self.assertIn("eyes_closed", keys)
        self.assertEqual(st.level, 2)

    def test_head_turned_away(self):
        e, t = calibrated_engine()
        st, t, keys = run(e, t, 2.5, yaw=40.0)
        self.assertIn("look_road_warn", keys)
        self.assertIn("look_road", keys)

    def test_short_glance_is_tolerated(self):
        e, t = calibrated_engine()
        _, t, keys = run(e, t, 0.6, yaw=40.0)       # mirror check
        self.assertEqual(keys, set())

    def test_gaze_away_with_head_forward(self):
        e, t = calibrated_engine()
        _, t, keys = run(e, t, 3.5, gaze_x=0.30)
        self.assertIn("gaze_away", keys)

    def test_frequent_yawning(self):
        e, t = calibrated_engine()
        allk = set()
        for _ in range(3):
            _, t, k = run(e, t, 1.2, mar=0.85)
            allk |= k
            _, t, k = run(e, t, 1.0)
            allk |= k
        self.assertIn("yawning", allk)

    def test_hand_near_face_then_phone(self):
        e, t = calibrated_engine()
        _, t, keys = run(e, t, 3.0, hand_near_face=True)
        self.assertIn("hand_face", keys)
        _, t, keys = run(e, t, 1.5, phone=True)
        self.assertIn("phone", keys)

    def test_driver_not_visible(self):
        e, t = calibrated_engine()
        _, t, keys = run(e, t, 3.0, face=False)
        self.assertIn("no_face", keys)

    def test_emergency_escalation(self):
        e, t = calibrated_engine()
        st, t, keys = run(e, t, 11.0, ear=0.05)
        self.assertIn("emergency", keys)
        self.assertEqual(st.level, 3)

    def test_recovers_after_alarm(self):
        e, t = calibrated_engine()
        _, t, _ = run(e, t, 2.0, ear=0.05)
        st, t, keys = run(e, t, 3.0)
        self.assertEqual(st.level, 0)

    def test_fatigue_index_rises_with_microsleeps(self):
        e, t = calibrated_engine()
        for _ in range(12):
            _, t, _ = run(e, t, 0.8, ear=0.05)      # micro-sleeps (0.5-1.2 s)
            _, t, _ = run(e, t, 1.2)
        self.assertGreater(e.fatigue, 20)


if __name__ == "__main__":
    unittest.main()
