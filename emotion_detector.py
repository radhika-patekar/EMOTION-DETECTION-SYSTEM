"""
Real-Time Facial Emotion Detection System  — Enhanced with Emotion Analysis Mode
═══════════════════════════════════════════════════════════════════════════════
KEYBOARD SHORTCUTS
  1–7  : Select target emotion to analyse (Angry/Disgust/Fear/Happy/Neutral/Sad/Surprise)
  0    : Clear target (return to standard mode)
  T    : Cycle alert threshold  (50% → 60% → 70% → 80%)
  R    : Reset target-emotion statistics
  S    : Save current frame as PNG screenshot
  Q / ESC : Quit
"""

import cv2
import numpy as np
import time
import os
import sys
from collections import deque
from datetime import datetime

# ─── Emotion Configuration ────────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    (0,   50,  220),
    'Disgust':  (0,  180,   60),
    'Fear':     (160,  0,  200),
    'Happy':    (0,  220,  220),
    'Neutral':  (200, 200, 200),
    'Sad':      (220, 100,   0),
    'Surprise': (0,  165,  255),
}

EMOTION_EMOJIS = {
    'Angry':    'Angry',
    'Disgust':  'Disgust',
    'Fear':     'Fear',
    'Happy':    'Happy',
    'Neutral':  'Neutral',
    'Sad':      'Sad',
    'Surprise': 'Surprise',
}

IMG_SIZE         = 48
GRAPH_HISTORY    = 120
ALERT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]

KEY_MAP = {
    ord('1'): 'Angry',   ord('2'): 'Disgust', ord('3'): 'Fear',
    ord('4'): 'Happy',   ord('5'): 'Neutral', ord('6'): 'Sad',
    ord('7'): 'Surprise',
}


# ─── Model Builder ────────────────────────────────────────────────────────────
def build_emotion_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv2D, MaxPooling2D, Dropout,
            BatchNormalization, GlobalAveragePooling2D, Dense
        )
        from tensorflow.keras.regularizers import l2

        model = Sequential([
            Conv2D(32, (3,3), padding='same', activation='relu',
                   input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2), Dropout(0.25),

            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2), Dropout(0.25),

            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2), Dropout(0.40),

            Conv2D(256, (3,3), padding='same', activation='relu'),
            BatchNormalization(), Dropout(0.40),

            GlobalAveragePooling2D(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(), Dropout(0.50),
            Dense(128, activation='relu'), Dropout(0.50),
            Dense(len(EMOTIONS), activation='softmax'),
        ], name='EmotionCNN')

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    except ImportError:
        print("[ERROR] TensorFlow not installed. pip install tensorflow")
        return None


# ─── Face Preprocessor ────────────────────────────────────────────────────────
def preprocess_face(face_img):
    gray       = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ─── Target Emotion Tracker ───────────────────────────────────────────────────
class EmotionTracker:
    """Tracks live statistics for a single chosen target emotion."""

    def __init__(self):
        self.target        = None          # str or None
        self.history       = deque(maxlen=GRAPH_HISTORY)
        self.peak_conf     = 0.0
        self.running_sum   = 0.0
        self.sample_count  = 0
        self.alert_idx     = 0
        self.alert_triggered  = False
        self.alert_cooldown   = 0
        self.detect_count  = 0
        self.start_ts      = time.time()

    @property
    def alert_threshold(self):
        return ALERT_THRESHOLDS[self.alert_idx]

    def cycle_threshold(self):
        self.alert_idx = (self.alert_idx + 1) % len(ALERT_THRESHOLDS)

    def reset(self):
        self.history.clear()
        self.peak_conf       = 0.0
        self.running_sum     = 0.0
        self.sample_count    = 0
        self.alert_triggered = False
        self.alert_cooldown  = 0
        self.detect_count    = 0
        self.start_ts        = time.time()

    def update(self, probabilities, dominant_emotion):
        if self.target is None:
            return
        idx  = EMOTIONS.index(self.target)
        conf = float(probabilities[idx])
        self.history.append(conf)
        self.running_sum  += conf
        self.sample_count += 1
        self.peak_conf     = max(self.peak_conf, conf)
        if dominant_emotion == self.target:
            self.detect_count += 1
        # alert gating
        if conf >= self.alert_threshold:
            self.alert_triggered = True
            self.alert_cooldown  = 45
        else:
            if self.alert_cooldown > 0:
                self.alert_cooldown -= 1
            else:
                self.alert_triggered = False

    @property
    def avg_conf(self):
        return self.running_sum / self.sample_count if self.sample_count else 0.0

    @property
    def current_conf(self):
        return self.history[-1] if self.history else 0.0

    @property
    def elapsed(self):
        s = int(time.time() - self.start_ts)
        return f"{s//60:02d}:{s%60:02d}"


# ─── Overlay Renderer ─────────────────────────────────────────────────────────
class OverlayRenderer:
    FONT       = cv2.FONT_HERSHEY_DUPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, w=1280, h=720):
        self.w = w
        self.h = h

    # helpers
    def _alpha_rect(self, frame, x1, y1, x2, y2, color, alpha=0.45):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    def _text(self, frame, txt, x, y, scale=0.55, color=(220,220,220),
              thick=1, font=None):
        cv2.putText(frame, txt, (x, y),
                    font or self.FONT_SMALL, scale, color, thick, cv2.LINE_AA)

    def _hline(self, frame, x1, x2, y, color=(50,80,110)):
        cv2.line(frame, (x1, y), (x2, y), color, 1)

    # ── face bounding box ────────────────────────────────────────────────────
    def draw_face_box(self, frame, x, y, w, h, emotion, confidence,
                      is_target=False, alert=False):
        color = EMOTION_COLORS.get(emotion, (255,255,255))

        # Pulsing ring when alert active
        if alert:
            alpha = 0.15 + 0.12 * abs(np.sin(time.time() * 6))
            self._alpha_rect(frame, x-8, y-8, x+w+8, y+h+8, color, alpha)

        blen  = max(20, w // 4)
        thick = 4 if is_target else 3
        segs = [
            ((x,       y),       (x+blen,   y)),
            ((x,       y),       (x,        y+blen)),
            ((x+w,     y),       (x+w-blen, y)),
            ((x+w,     y),       (x+w,      y+blen)),
            ((x,       y+h),     (x+blen,   y+h)),
            ((x,       y+h),     (x,        y+h-blen)),
            ((x+w,     y+h),     (x+w-blen, y+h)),
            ((x+w,     y+h),     (x+w,      y+h-blen)),
        ]
        for p1, p2 in segs:
            cv2.line(frame, p1, p2, color, thick)

        if is_target:
            cv2.rectangle(frame, (x+2, y+2), (x+w-2, y+h-2), color, 1)

        label = f"{emotion}  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.7, 2)
        self._alpha_rect(frame, x-6, y-th-18, x+tw+4, y-8, (20,20,20), 0.7)
        cv2.putText(frame, label, (x, y-12), self.FONT, 0.7, color, 2, cv2.LINE_AA)

    # ── header bar ───────────────────────────────────────────────────────────
    def draw_header(self, frame, target):
        self._alpha_rect(frame, 0, 0, self.w-260, 44, (5,8,20), 0.75)
        if target:
            hint = (f"ANALYSING: {target.upper()}  |  "
                    f"1-7=emotion  0=clear  T=threshold  R=reset  S=save  Q=quit")
        else:
            hint = ("Select emotion: 1=Angry  2=Disgust  3=Fear  4=Happy  "
                    "5=Neutral  6=Sad  7=Surprise  |  Q=quit")
        self._text(frame, hint, 16, 28, 0.46, (80,140,200))

    # ── right sidebar ────────────────────────────────────────────────────────
    def draw_sidebar(self, frame, emotion, confidence, probabilities,
                     fps, face_count, tracker):
        sw = 260
        self._alpha_rect(frame, self.w-sw, 0, self.w, self.h, (10,10,18), 0.85)

        y = 36
        self._text(frame, "EMOTION AI", self.w-sw+14, y,
                   0.75, (80,220,255), 2, self.FONT)
        y += 6
        self._hline(frame, self.w-sw+8, self.w-8, y, (60,100,140))

        if emotion:
            y += 26
            c = EMOTION_COLORS.get(emotion, (200,200,200))
            self._alpha_rect(frame, self.w-sw+8, y-18, self.w-8, y+40, c, 0.25)
            cv2.rectangle(frame, (self.w-sw+8, y-18), (self.w-8, y+40), c, 1)
            cv2.putText(frame, emotion.upper(), (self.w-sw+18, y+14),
                        self.FONT, 1.0, c, 2, cv2.LINE_AA)
            y += 34
            self._text(frame, f"Confidence: {confidence:.1%}",
                       self.w-sw+18, y, 0.52, (180,180,180))

        y += 22
        self._hline(frame, self.w-sw+8, self.w-8, y)
        y += 20
        self._text(frame, "ALL EMOTIONS", self.w-sw+14, y, 0.40, (100,150,200))
        y += 10

        bar_w  = sw - 36
        in_tgt = (tracker.target is not None)

        for emo, prob in zip(EMOTIONS, probabilities):
            y += 24
            c      = EMOTION_COLORS.get(emo, (200,200,200))
            is_tgt = (emo == tracker.target)
            dim    = not in_tgt or is_tgt
            bc     = c if dim else tuple(int(v * 0.3) for v in c)
            lc     = (220,220,220) if dim else (90,90,100)

            cv2.rectangle(frame, (self.w-sw+14, y-10),
                          (self.w-sw+14+bar_w, y+4), (40,40,55), -1)
            filled = int(bar_w * prob)
            if filled > 0:
                cv2.rectangle(frame, (self.w-sw+14, y-10),
                              (self.w-sw+14+filled, y+4), bc, -1)
            if is_tgt:
                cv2.rectangle(frame, (self.w-sw+14, y-10),
                              (self.w-sw+14+bar_w, y+4), c, 1)

            self._text(frame, f"{emo[:7]:<7} {prob:.0%}",
                       self.w-sw+16, y+1, 0.42, lc)

        y = self.h - 60
        self._hline(frame, self.w-sw+8, self.w-8, y)
        y += 18
        self._text(frame, f"FPS: {fps:.1f}",     self.w-sw+14, y, 0.5, (100,200,120))
        y += 18
        self._text(frame, f"Faces: {face_count}", self.w-sw+14, y, 0.5, (100,200,120))

    # ── bottom analysis panel ────────────────────────────────────────────────
    def draw_analysis_panel(self, frame, tracker):
        if tracker.target is None:
            return

        emo   = tracker.target
        color = EMOTION_COLORS.get(emo, (200,200,200))
        hist  = list(tracker.history)

        # Panel geometry
        px, py = 10, self.h - 195
        pw, ph = self.w - 280, 185

        # Background + border
        self._alpha_rect(frame, px, py, px+pw, py+ph, (8,12,25), 0.88)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 1)

        # Title strip (solid)
        cv2.rectangle(frame, (px, py), (px+pw, py+26), color, -1)
        title = (f"  ANALYSING: {emo.upper()}   |   "
                 f"Alert threshold: {tracker.alert_threshold:.0%}   |   "
                 f"Elapsed: {tracker.elapsed}")
        self._text(frame, title, px+6, py+17, 0.46, (10,10,20), 1)

        # ── stat cards ───────────────────────────────────────────────────────
        stats = [
            ("Current",  f"{tracker.current_conf:.1%}"),
            ("Peak",     f"{tracker.peak_conf:.1%}"),
            ("Average",  f"{tracker.avg_conf:.1%}"),
            ("Detected", f"{tracker.detect_count}x"),
            ("Samples",  f"{tracker.sample_count}"),
        ]
        card_w, card_h = 90, 44
        cx = px + 8
        for label, val in stats:
            self._alpha_rect(frame, cx, py+32, cx+card_w, py+32+card_h,
                             (18,24,42), 0.9)
            cv2.rectangle(frame, (cx, py+32), (cx+card_w, py+32+card_h),
                          (60,80,110), 1)
            self._text(frame, val,   cx+6, py+56, 0.65, color, 1, self.FONT)
            self._text(frame, label, cx+6, py+71, 0.38, (120,145,175))
            cx += card_w + 6

        # ── sparkline graph ──────────────────────────────────────────────────
        gx = px + 8
        gy = py + 86
        gw = pw - 16
        gh = ph - 96

        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (16,20,36), -1)

        # Grid lines at 25 / 50 / 75%
        for lvl in (0.25, 0.50, 0.75):
            ly = gy + gh - int(lvl * gh)
            cv2.line(frame, (gx, ly), (gx+gw, ly), (38,50,70), 1)
            self._text(frame, f"{int(lvl*100)}%", gx+3, ly-2, 0.32, (65,88,120))

        # Alert threshold line
        ty = gy + gh - int(tracker.alert_threshold * gh)
        cv2.line(frame, (gx, ty), (gx+gw, ty), (0,200,120), 1)
        self._text(frame, f"alert {tracker.alert_threshold:.0%}",
                   gx+gw-70, ty-3, 0.34, (0,200,120))

        # Confidence sparkline
        if len(hist) >= 2:
            pts = []
            for i, v in enumerate(hist):
                sx = gx + int(i / max(len(hist)-1, 1) * gw)
                sy = gy + gh - int(np.clip(v, 0, 1) * gh)
                pts.append((sx, sy))

            # Filled area
            poly = [(gx, gy+gh)] + pts + [(pts[-1][0], gy+gh)]
            ov2  = frame.copy()
            cv2.fillPoly(ov2, [np.array(poly, dtype=np.int32)], color)
            cv2.addWeighted(ov2, 0.18, frame, 0.82, 0, frame)

            # Line
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], color, 2)

            # Current-value dot
            cv2.circle(frame, pts[-1], 4, color, -1)
            cv2.circle(frame, pts[-1], 6, (255,255,255), 1)

        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (48,68,100), 1)

    # ── full-screen alert flash ───────────────────────────────────────────────
    def draw_alert(self, frame, tracker):
        if not tracker.alert_triggered or tracker.target is None:
            return
        color = EMOTION_COLORS.get(tracker.target, (255,255,255))
        alpha = 0.06 + 0.06 * abs(np.sin(time.time() * 5))
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (self.w, self.h), color, 14)
        cv2.addWeighted(ov, alpha*2, frame, 1-alpha*2, 0, frame)

        badge = f"  ! {tracker.target.upper()} DETECTED  {tracker.current_conf:.0%}  !"
        (bw, bh), _ = cv2.getTextSize(badge, self.FONT, 0.75, 2)
        bx = (self.w - 260 - bw) // 2
        by = 64
        self._alpha_rect(frame, bx-8, by-bh-6, bx+bw+8, by+6, color, 0.85)
        cv2.putText(frame, badge, (bx, by),
                    self.FONT, 0.75, (10,10,10), 2, cv2.LINE_AA)

    def draw_no_face(self, frame):
        cx = (self.w - 260) // 2
        cv2.putText(frame, "No Face Detected", (cx-130, self.h//2),
                    self.FONT, 0.85, (60,80,120), 2, cv2.LINE_AA)

    def draw_key_hint(self, frame, tracker):
        if tracker.target:
            return
        labels = [f"{i+1}={e[0]}" for i, e in enumerate(EMOTIONS)]
        self._text(frame, "Select emotion: " + "  ".join(labels),
                   16, self.h-16, 0.40, (60,90,130))


# ─── Main Detector ────────────────────────────────────────────────────────────
class EmotionDetector:

    def __init__(self, model_path=None, camera_index=0):
        self.camera_index = camera_index
        self.model        = None
        self.face_cascade = None
        self.fps_buffer   = deque(maxlen=30)
        self.tracker      = EmotionTracker()
        self._load_resources(model_path)

    def _load_resources(self, model_path):
        cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade)
        if self.face_cascade.empty():
            raise RuntimeError(f"Cannot load cascade: {cascade}")
        print("[OK] Haar cascade loaded")

        self.model = build_emotion_model()
        if self.model is None:
            return

        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"[OK] Weights loaded: {model_path}")
        else:
            print("[INFO] No weights — running in demo (random) mode.")
            print("       Train first:  python train_model.py")

    def _predict(self, face_img):
        if self.model is None:
            probs = np.random.dirichlet(np.ones(len(EMOTIONS)) * 0.5)
            idx   = int(np.argmax(probs))
            return EMOTIONS[idx], float(probs[idx]), probs.tolist()
        tensor = preprocess_face(face_img)
        probs  = self.model.predict(tensor, verbose=0)[0]
        idx    = int(np.argmax(probs))
        return EMOTIONS[idx], float(probs[idx]), probs.tolist()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[OK] Camera {self.camera_index}  {actual_w}x{actual_h}")
        print("     1-7=emotion  0=clear  T=threshold  R=reset  S=save  Q=quit")

        renderer = OverlayRenderer(actual_w, actual_h)
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Detection', actual_w, actual_h)

        tracker = self.tracker
        t_prev  = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS
            now = time.time()
            self.fps_buffer.append(1.0 / max(now - t_prev, 1e-6))
            fps    = float(np.mean(self.fps_buffer))
            t_prev = now

            # Detect faces
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

            dominant_emotion, dominant_conf = None, 0.0
            dominant_probs = [1.0/len(EMOTIONS)] * len(EMOTIONS)

            for (x, y, w, h) in faces:
                emotion, conf, probs = self._predict(frame[y:y+h, x:x+w])
                is_tgt = (emotion == tracker.target)
                renderer.draw_face_box(
                    frame, x, y, w, h, emotion, conf,
                    is_target=is_tgt,
                    alert=(is_tgt and tracker.alert_triggered),
                )
                if conf > dominant_conf:
                    dominant_emotion, dominant_conf = emotion, conf
                    dominant_probs = probs

            # Update tracker
            if len(faces) > 0:
                tracker.update(dominant_probs, dominant_emotion)
            elif tracker.target is not None:
                tracker.history.append(0.0)

            # Draw UI layers
            renderer.draw_header(frame, tracker.target)
            renderer.draw_sidebar(frame, dominant_emotion, dominant_conf,
                                  dominant_probs, fps, len(faces), tracker)
            renderer.draw_analysis_panel(frame, tracker)
            renderer.draw_alert(frame, tracker)
            if len(faces) == 0:
                renderer.draw_no_face(frame)
            renderer.draw_key_hint(frame, tracker)

            cv2.imshow('Emotion Detection', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in KEY_MAP:
                new = KEY_MAP[key]
                if tracker.target != new:
                    tracker.target = new
                    tracker.reset()
                    print(f"[TARGET] Analysing: {new}")
            elif key == ord('0'):
                tracker.target = None
                tracker.reset()
                print("[TARGET] Cleared")
            elif key in (ord('t'), ord('T')):
                tracker.cycle_threshold()
                print(f"[THRESHOLD] {tracker.alert_threshold:.0%}")
            elif key in (ord('r'), ord('R')):
                tracker.reset()
                print(f"[RESET] Stats cleared for {tracker.target}")
            elif key in (ord('s'), ord('S')):
                fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame)
                print(f"[SAVE] {fname}")

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model_path   = 'emotion_model_weights.h5'
    camera_idx   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_target = sys.argv[2].capitalize() if len(sys.argv) > 2 else None

    if start_target and start_target not in EMOTIONS:
        print(f"[WARN] '{start_target}' invalid. Choose from: {EMOTIONS}")
        start_target = None

    detector = EmotionDetector(model_path=model_path, camera_index=camera_idx)

    if start_target:
        detector.tracker.target = start_target
        print(f"[TARGET] Pre-selected: {start_target}")

    detector.run()
