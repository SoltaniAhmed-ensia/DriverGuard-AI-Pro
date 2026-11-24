"""Audio alarms (generated in code - no sound files needed) and optional voice."""
import queue
import threading

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


def _tone(freq, dur, vol=0.7, sr=44100):
    t = np.linspace(0, dur, int(sr * dur), False)
    return np.sin(2 * np.pi * freq * t) * vol


def _snd(wave):
    return pygame.sndarray.make_sound((wave * 32767).astype(np.int16))


class Alarm:
    """Level 1: short beep. Level 2: siren. Level 3: fast emergency siren."""

    def __init__(self):
        self.ok = False
        self.muted = False
        self.loop = 0          # which looping sound is playing (0 none, 2, 3)
        self.last_beep = 0.0
        if pygame is None:
            print("[WARN] pygame missing - no sound")
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1,
                              buffer=512, allowedchanges=0)
            self.beep = _snd(_tone(900, 0.25, 0.5))
            self.siren = _snd(np.concatenate([_tone(880, 0.22), _tone(1250, 0.22)] * 2))
            self.emerg = _snd(np.concatenate([_tone(700, 0.11, 0.9), _tone(1600, 0.11, 0.9)] * 4))
            self.ok = True
        except Exception as e:
            print("[WARN] Audio unavailable:", e)

    def update(self, level, now):
        if not self.ok or self.muted:
            self.stop()
            return
        want = level if level >= 2 else 0
        if want != self.loop:
            self.stop()
            if want == 2:
                self.siren.play(loops=-1)
            elif want == 3:
                self.emerg.play(loops=-1)
            self.loop = want
        if level == 1 and now - self.last_beep > 1.5:
            self.beep.play()
            self.last_beep = now

    def stop(self):
        if self.ok and self.loop:
            (self.siren if self.loop == 2 else self.emerg).stop()
        self.loop = 0


class Voice:
    """Optional offline text-to-speech (pyttsx3) running in its own thread."""

    def __init__(self, enabled, phrases):
        self.ok = False
        self.phrases = phrases or {}
        self.cool = {}
        if not enabled:
            return
        try:
            import pyttsx3  # noqa: F401
        except Exception:
            print("[WARN] pyttsx3 not installed - voice disabled (pip install pyttsx3)")
            return
        self.q = queue.Queue(maxsize=2)
        self.ok = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import pyttsx3
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", 165)
        except Exception as e:
            print("[WARN] TTS init failed:", e)
            return
        while True:
            txt = self.q.get()
            try:
                eng.say(txt)
                eng.runAndWait()
            except Exception:
                pass

    def say(self, key, now, cooldown=6.0):
        if not self.ok or key not in self.phrases:
            return
        if now - self.cool.get(key, -1e9) < cooldown:
            return
        self.cool[key] = now
        try:
            self.q.put_nowait(self.phrases[key])
        except queue.Full:
            pass
