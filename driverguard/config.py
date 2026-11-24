from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- calibration ---
    calib_seconds: float = 4.0       # driver looks straight ahead
    # --- eyes / drowsiness ---
    ear_ratio: float = 0.72          # closed if EAR < ratio * personal open-eye EAR
    microsleep_s: float = 0.5        # closure longer than this = micro-sleep
    closed_alarm_s: float = 1.2      # closure longer than this = loud alarm
    perclos_limit: float = 0.30      # PERCLOS > 30% over window = fatigue
    perclos_window_s: float = 30.0
    # --- yawning ---
    mar_yawn: float = 0.60
    yawn_min_s: float = 0.8
    yawns_limit: int = 3             # yawns per 60 s
    # --- head pose ---
    yaw_limit: float = 25.0          # degrees from calibrated forward
    pitch_limit: float = 20.0
    away_warn_s: float = 1.0
    away_alarm_s: float = 2.0
    # --- gaze (iris tracking) ---
    gaze_x_limit: float = 0.17       # fraction of eye width
    gaze_y_limit: float = 0.22
    gaze_warn_s: float = 1.5
    gaze_alarm_s: float = 3.0
    # --- hands / phone ---
    hand_face_s: float = 2.5
    phone_alarm_s: float = 1.0
    # --- presence ---
    noface_alarm_s: float = 2.0
    # --- fatigue index / emergency ---
    fatigue_warn: float = 60.0
    emergency_s: float = 8.0         # level-2 alarm this long = driver unresponsive
    # --- night mode ---
    dark_brightness: float = 80.0    # mean gray below this -> CLAHE enhancement
    # --- logging ---
    log_interval_s: float = 0.5

    def as_dict(self):
        return asdict(self)
