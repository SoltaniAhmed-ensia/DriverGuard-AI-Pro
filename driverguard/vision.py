"""Computer-vision layer: face mesh, eye/mouth ratios, head pose, iris gaze,
hands near face, optional YOLO phone detector, low-light enhancement."""
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp

R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_RING = [78, 82, 312, 308, 317, 87]
MOUTH_V = [(13, 14), (82, 87), (312, 317)]
MOUTH_H = (78, 308)
POSE_IDS = [1, 152, 33, 263, 61, 291]
# iris centres (needs refine_landmarks=True -> 478 points)
R_IRIS, L_IRIS = 468, 473
R_CORNERS = (33, 133)    # outer, inner
L_CORNERS = (263, 362)

MODEL_3D = np.array([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                     (225, 170, -135), (-150, -150, -125), (150, -150, -125)],
                    dtype=np.float64)


@dataclass
class Metrics:
    face: bool = False
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    gaze_ok: bool = False
    hand_near_face: bool = False
    phone: bool = False
    points: object = None          # (468, 2) pixel landmarks, for drawing
    iris: object = None            # two iris centres in pixels
    hand_pts: list = field(default_factory=list)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def eye_ratio(p, idx):
    a, b, c, d, e, f = [p[i] for i in idx]
    return (_d(b, f) + _d(c, e)) / (2.0 * _d(a, d) + 1e-6)


def mouth_ratio(p):
    v = np.mean([_d(p[a], p[b]) for a, b in MOUTH_V])
    return v / (_d(p[MOUTH_H[0]], p[MOUTH_H[1]]) + 1e-6)


def _norm_angle(a):
    a = (a + 180) % 360 - 180
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


def head_pose(p, w, h):
    img = p[POSE_IDS].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    ang = cv2.RQDecomp3x3(rot)[0]
    return _norm_angle(ang[0]), _norm_angle(ang[1])    # pitch, yaw


def iris_gaze(p_all):
    """Iris offset from the eye centre, as a fraction of eye width.
    Returns (gx, gy, iris_points) - image-right / image-down positive."""
    gx, gy, pts = [], [], []
    for iris, (outer, inner), up, low in ((R_IRIS, R_CORNERS, 159, 145),
                                          (L_IRIS, L_CORNERS, 386, 374)):
        o, i, c = p_all[outer], p_all[inner], p_all[iris]
        width = abs(i[0] - o[0]) + 1e-6
        centre = (o + i) / 2.0
        gx.append((c[0] - centre[0]) / width)
        mid_y = (p_all[up][1] + p_all[low][1]) / 2.0
        gy.append((c[1] - mid_y) / width)
        pts.append(c)
    return float(np.mean(gx)), float(np.mean(gy)), pts


def enhance_low_light(frame, threshold):
    """CLAHE on the luminance channel when the scene is dark. Returns (frame, dark?)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= threshold:
        return frame, False
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 0.7
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, table), True


class FaceAnalyzer:
    def __init__(self, use_hands=True):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = None
        if use_hands:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=2, min_detection_confidence=0.6,
                min_tracking_confidence=0.5)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        m = Metrics()
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return m
        lm = res.multi_face_landmarks[0].landmark
        n = len(lm)
        p_all = np.array([(l.x * w, l.y * h) for l in lm])
        p = p_all[:468]
        m.face = True
        m.points = p
        m.ear = (eye_ratio(p, R_EYE) + eye_ratio(p, L_EYE)) / 2.0
        m.mar = mouth_ratio(p)
        pose = head_pose(p, w, h)
        if pose:
            m.pitch, m.yaw = pose
        if n >= 478:
            m.gaze_x, m.gaze_y, m.iris = iris_gaze(p_all)
            m.gaze_ok = True

        if self.hands is not None:
            hres = self.hands.process(rgb)
            if hres.multi_hand_landmarks:
                x0, y0 = p.min(axis=0)
                x1, y1 = p.max(axis=0)
                fw, fh = x1 - x0, y1 - y0
                box = (x0 - 0.5 * fw, y0 - 0.15 * fh, x1 + 0.5 * fw, y1 + 0.30 * fh)
                for hl in hres.multi_hand_landmarks:
                    pts = np.array([(q.x * w, q.y * h) for q in hl.landmark])
                    m.hand_pts.append(pts)
                    inside = np.sum((pts[:, 0] > box[0]) & (pts[:, 0] < box[2]) &
                                    (pts[:, 1] > box[1]) & (pts[:, 1] < box[3]))
                    if inside >= 5:
                        m.hand_near_face = True
        return m


class PhoneDetector:
    """YOLOv8-nano 'cell phone' detector (COCO class 67). Runs every N frames."""

    def __init__(self, every=3, conf=0.40):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.every, self.conf = every, conf
        self.i = 0
        self.boxes = []
        self.last_seen = 0.0

    def update(self, frame, now):
        self.i += 1
        if self.i % self.every == 0:
            r = self.model.predict(frame, classes=[67], conf=self.conf, verbose=False)[0]
            self.boxes = [tuple(map(int, b.xyxy[0])) for b in r.boxes]
            if self.boxes:
                self.last_seen = now
        active = (now - self.last_seen) < 0.8
        return active, (self.boxes if active else [])
