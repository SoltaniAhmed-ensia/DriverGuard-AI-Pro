"""On-screen display. Text is drawn with Pillow so Arabic and French accents work."""
import os
import unicodedata

import cv2
import numpy as np

GREEN, YELLOW, RED, CYAN = (80, 200, 80), (0, 210, 255), (60, 60, 240), (255, 200, 0)
WHITE, GREY = (255, 255, 255), (170, 170, 170)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class TextLayer:
    """Collects text items during a frame and renders them all at once."""

    def __init__(self, lang):
        self.lang = lang
        self.font_path = find_font()
        self.fonts = {}
        self.items = []
        self.shaper = None
        self.available = self.font_path is not None
        if lang == "ar" and self.available:
            try:
                import arabic_reshaper
                try:
                    from bidi import get_display
                except ImportError:
                    from bidi.algorithm import get_display
                self.shaper = (arabic_reshaper, get_display)
            except Exception:
                print("[WARN] Arabic needs: pip install arabic-reshaper python-bidi")
                self.available = False

    def _shape(self, text):
        if self.shaper:
            return self.shaper[1](self.shaper[0].reshape(text))
        return text

    def add(self, text, x, y, size=18, color=WHITE, anchor="l"):
        self.items.append((text, x, y, size, color, anchor))

    def render(self, img):
        if not self.items:
            return img
        if not self.available:                      # fallback: Hershey font, ASCII only
            for text, x, y, size, color, anchor in self.items:
                t = _ascii(text)
                scale = size / 32.0
                (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                if anchor == "m":
                    x -= tw // 2
                elif anchor == "r":
                    x -= tw
                cv2.putText(img, t, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, t, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 1, cv2.LINE_AA)
            self.items = []
            return img
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, x, y, size, color, anchor in self.items:
            font = self.fonts.get(size)
            if font is None:
                font = self.fonts[size] = ImageFont.truetype(
                    self.font_path, size, layout_engine=ImageFont.Layout.BASIC)
            rgb = (color[2], color[1], color[0])
            draw.text((x, y), self._shape(text), font=font, fill=rgb,
                      anchor={"l": "la", "m": "ma", "r": "ra"}[anchor],
                      stroke_width=2, stroke_fill=(0, 0, 0))
        self.items = []
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def panel(img, x0, y0, x1, y1, alpha=0.55, color=(20, 20, 20)):
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    roi = img[y0:y1, x0:x1]
    over = np.full_like(roi, color)
    img[y0:y1, x0:x1] = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)


def bar(img, x, y, w, h, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0, min(1, value))), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (110, 110, 110), 1)


def level_color(level):
    return RED if level >= 2 else YELLOW if level == 1 else GREEN


def draw_overlays(frame, m, st):
    """Landmarks / boxes drawn directly on the frame (no text)."""
    col = level_color(st.level) if st.calibrated else CYAN
    if m.face and m.points is not None:
        p = m.points
        x0, y0 = p.min(axis=0).astype(int)
        x1, y1 = p.max(axis=0).astype(int)
        L = max(18, (x1 - x0) // 6)
        for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (ax, ay), (ax + dx * L, ay), col, 3)
            cv2.line(frame, (ax, ay), (ax, ay + dy * L), col, 3)
        from .vision import R_EYE, L_EYE, MOUTH_RING
        for idx in (R_EYE, L_EYE, MOUTH_RING):
            cv2.polylines(frame, [p[idx].astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
        if m.iris:
            for c in m.iris:
                cv2.circle(frame, (int(c[0]), int(c[1])), 3, (0, 255, 255), -1)
    for hp in m.hand_pts:
        hc = RED if (st.hand_d > 0) else (200, 200, 200)
        x0, y0 = hp.min(axis=0).astype(int)
        x1, y1 = hp.max(axis=0).astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), hc, 2)


def draw_hud(frame, st, m, tr, layer, ctx):
    """ctx: dict(fps, muted, dark, gt, log, hist, now, phone_boxes, calib_left)"""
    h, w = frame.shape[:2]
    big = max(24, w // 42)
    sm = max(15, w // 80)
    lvl = st.level
    col = level_color(lvl)

    # flashing border on alarms
    if lvl >= 2 and int(ctx["now"] * 4) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 14)

    # ---- top banner ----
    emergency = lvl >= 3
    bh = int(big * 2.6) if emergency else int(big * 1.9)
    panel(frame, 0, 0, w, bh, 0.75, (0, 0, 120) if emergency else (20, 20, 20))
    if not st.calibrated:
        if st.face:
            left = max(0, round((1 - st.calib_progress) * ctx["calib_s"]))
            txt = tr["calibrating"].format(s=left)
        else:
            txt = tr["show_face"]
        layer.add(txt, w // 2, int(big * 0.4), big, YELLOW, "m")
    elif st.alerts:
        top = max(st.alerts, key=lambda a: a.level)
        layer.add(tr[top.key], w // 2, int(big * 0.4), big, col, "m")
        if emergency:
            layer.add(tr["emergency_sub"], w // 2, int(big * 1.5), sm + 2, WHITE, "m")
    else:
        layer.add(tr["ok"], w // 2, int(big * 0.4), big, GREEN, "m")
    layer.add(tr["title"], 14, 8, sm, GREY, "l")

    # ---- right metrics panel ----
    pw = int(w * 0.25)
    px = w - pw - 10
    py = bh + 10
    rows = [
        (tr["perclos_l"], f"{st.perclos * 100:.0f}%"),
        (tr["blinks"], f"{st.blink_rate:.0f}"),
        (tr["yawns"], f"{st.yawns}"),
        (tr["micro"], f"{st.micro}"),
        (tr["closed"], f"{st.closed_d:.1f}s"),
        (tr["away"], f"{st.away_d:.1f}s"),
        (tr["gaze"], f"{st.gaze_d:.1f}s"),
        ("EAR / MAR", f"{st.ear:.2f} / {st.mar:.2f}"),
        ("Yaw / Pitch", f"{int(round(st.dyaw)):+d} / {int(round(st.dpitch)):+d}"),
        ("FPS", f"{ctx['fps']:.0f}"),
    ]
    rh = sm + 9
    ph = 2 * (rh + 14) + len(rows) * rh + 14
    panel(frame, px - 8, py - 6, px + pw + 4, py + ph, 0.6)
    y = py
    for label, val, v01, c in (
            (tr["attention"], f"{st.attention:.0f}%", st.attention / 100,
             GREEN if st.attention > 70 else YELLOW if st.attention > 40 else RED),
            (tr["fatigue_idx"], f"{st.fatigue:.0f}%", st.fatigue / 100,
             GREEN if st.fatigue < 30 else YELLOW if st.fatigue < 60 else RED)):
        layer.add(label, px, y, sm, WHITE, "l")
        layer.add(val, px + pw - 8, y, sm, WHITE, "r")
        bar(frame, px, y + sm + 6, pw - 8, 10, v01, c)
        y += rh + 14
    for label, val in rows:
        layer.add(label, px, y, sm - 1, GREY, "l")
        layer.add(val, px + pw - 8, y, sm - 1, WHITE, "r")
        y += rh

    # ---- bottom-left: attention sparkline ----
    hist = ctx["hist"]
    sw, sh = int(w * 0.26), int(h * 0.12)
    sx, sy = 12, h - sh - 12
    panel(frame, sx - 6, sy - 24, sx + sw + 6, sy + sh + 6, 0.6)
    layer.add(f"{tr['attention']} - 60s", sx, sy - 22, sm - 2, GREY, "l")
    if len(hist) > 2:
        pts = np.array([(sx + int(i * sw / max(len(hist) - 1, 1)),
                         sy + sh - int(v / 100 * sh)) for i, v in enumerate(hist)],
                       np.int32)
        cv2.polylines(frame, [pts], False, CYAN, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (90, 90, 90), 1)

    # ---- bottom-right: recent alerts ----
    log = ctx["log"][-4:]
    if log:
        lw = int(w * 0.30)
        lx = w - lw - 10
        ly = h - (len(log) * (sm + 6)) - 40
        panel(frame, lx - 6, ly - 4, lx + lw + 4, h - 10, 0.6)
        layer.add(tr["alerts_log"], lx, ly - 2, sm - 2, GREY, "l")
        for i, (t, key, lv) in enumerate(log):
            layer.add(f"{int(t // 60):02d}:{int(t % 60):02d}  {tr[key]}", lx,
                      ly + 18 + i * (sm + 6), sm - 3, level_color(lv), "l")

    # ---- small flags ----
    yy = bh + 10
    if ctx["dark"]:
        layer.add(tr["low_light"], 14, yy, sm, YELLOW, "l")
        yy += sm + 8
    active_gt = [tr["cats"][c] for c, on in ctx["gt"].items() if on]
    if active_gt:
        cv2.circle(frame, (24, yy + sm // 2 + 2), 7, RED, -1)
        layer.add(f"{tr['gt']}: " + ", ".join(active_gt), 38, yy, sm, RED, "l")
        yy += sm + 8
    if ctx["muted"]:
        layer.add(tr["muted"], 14, yy, sm, YELLOW, "l")

    for (x1, y1, x2, y2) in ctx["phone_boxes"]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3)
        layer.add("PHONE", x1, max(y1 - sm - 6, 0), sm, RED, "l")
    return layer.render(frame)
