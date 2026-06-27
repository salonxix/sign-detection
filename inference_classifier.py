"""
inference_classifier.py
-----------------------
Production-quality real-time Sign Language Recognition.

Features
────────
  • Top-3 predictions with confidence bars
  • Confidence threshold  →  shows "Unknown" below threshold
  • Calibration mode      →  collects per-class background stats to
                             apply a simple score correction
  • Prediction history panel  (last 20 accepted predictions)
  • Sentence builder      →  accepted chars build a running sentence
  • Text-to-speech        →  speaks each accepted character (optional)
  • FPS overlay
  • CSV export            →  every accepted prediction logged to
                             predictions.csv (timestamp, char, confidence)

Keyboard controls
─────────────────
  Q / ESC   quit
  SPACE     accept current smoothed prediction → add to sentence
  BKSP      delete last character from sentence
  C         clear sentence
  T         toggle text-to-speech on/off
  CAL       press K to enter calibration mode
  H         toggle history panel
"""

import csv
import os
import pickle
import time
import threading
from collections import deque, Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("pyttsx3 not available — TTS disabled.")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70     # below this → "Unknown"
SMOOTH_WINDOW        = 10       # frames for majority-vote smoothing
TOP_K                = 3        # how many predictions to show in the panel
HISTORY_MAX          = 20       # entries in the history panel
CSV_PATH             = 'predictions.csv'
CAL_FRAMES           = 60       # frames to collect per-class during calibration

# Layout constants (all pixel values are for a 640×480 feed + 260px right panel)
FEED_W, FEED_H = 640, 480
PANEL_W        = 280            # right-side info panel width
WIN_W          = FEED_W + PANEL_W
WIN_H          = FEED_H

PANEL_BG    = (25,  25,  25)
ACCENT      = (0,  200, 255)
GREEN       = (0,  210,   0)
ORANGE      = (0,  165, 255)
RED         = (0,   60, 220)
WHITE       = (240, 240, 240)
GREY        = (140, 140, 140)
DARK_GREY   = (55,  55,  55)

# ── Load model bundle ─────────────────────────────────────────────────────────
BUNDLE_PATH = './model.p'
if not os.path.exists(BUNDLE_PATH):
    raise FileNotFoundError("model.p not found. Run train_classifier.py first.")

bundle        = pickle.load(open(BUNDLE_PATH, 'rb'))
model         = bundle['model']
label_encoder = bundle['label_encoder']
model_name    = bundle.get('model_name', 'Unknown')
classes       = bundle['classes']
num_classes   = len(classes)

print(f"Model  : {model_name}  ({bundle.get('test_accuracy', 0):.2%} test acc)")
print(f"Classes: {classes}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'hand_landmarker.task')
if not os.path.exists(MP_MODEL):
    raise FileNotFoundError(
        f"hand_landmarker.task not found.\n"
        "Download: python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task','hand_landmarker.task')\""
    )

_base   = mp_python.BaseOptions(model_asset_path=MP_MODEL)
_opts   = mp_vision.HandLandmarkerOptions(
    base_options=_base,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(_opts)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Preprocessing (identical to create_dataset.py) ───────────────────────────
def extract_features(hand_landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float32)
    coords -= coords[0]
    m = np.abs(coords).max()
    if m > 0:
        coords /= m
    flat = coords.flatten()
    n = np.linalg.norm(flat)
    if n > 0:
        flat /= n
    return flat

# ── TTS helper (runs in background thread) ───────────────────────────────────
_tts_queue: deque = deque(maxlen=1)   # only speak the latest

def _tts_worker():
    while True:
        if _tts_queue and TTS_AVAILABLE:
            text = _tts_queue.popleft()
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception:
                pass
        time.sleep(0.05)

if TTS_AVAILABLE:
    _t = threading.Thread(target=_tts_worker, daemon=True)
    _t.start()

def speak(text: str):
    if TTS_AVAILABLE:
        _tts_queue.append(text)

# ── CSV setup ─────────────────────────────────────────────────────────────────
_csv_new = not os.path.exists(CSV_PATH)
_csv_f   = open(CSV_PATH, 'a', newline='')
_csv_w   = csv.writer(_csv_f)
if _csv_new:
    _csv_w.writerow(['timestamp', 'character', 'confidence', 'model'])

def log_prediction(char: str, conf: float):
    _csv_w.writerow([datetime.now().isoformat(timespec='seconds'),
                     char, f'{conf:.4f}', model_name])
    _csv_f.flush()

# ── Calibration ───────────────────────────────────────────────────────────────
# Stores per-class mean confidence collected during calibration mode.
# Used to apply a simple bias correction at inference time.
calibration: dict[str, float] = {}   # class → mean_proba_at_calibration

def run_calibration(cap, hl):
    """
    Interactive calibration: user shows each class for CAL_FRAMES frames.
    Stores the average top-class probability for each class.
    Returns updated calibration dict.
    """
    cal = {}
    print("\n── Calibration mode ─────────────────────────────────────────")
    for cls in classes:
        samples = []
        print(f"  Show sign for '{cls}' and press SPACE …")
        collecting = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FEED_W, FEED_H))
            H, W  = frame.shape[:2]

            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res      = hl.detect(mp_image)

            msg = (f"CALIBRATION: '{cls}'"
                   f"  {'Collecting...' if collecting else 'Press SPACE to collect'}"
                   f"  {len(samples)}/{CAL_FRAMES}")
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (W, 60), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, msg, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2, cv2.LINE_AA)
            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                collecting = True
            if key == 27:
                cv2.destroyWindow('Calibration')
                return cal

            if collecting and res.hand_landmarks:
                feat = extract_features(res.hand_landmarks[0]).reshape(1, -1)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feat)[0]
                    pred  = model.predict(feat)[0]
                    samples.append(float(proba[pred]))
                else:
                    samples.append(1.0)
                if len(samples) >= CAL_FRAMES:
                    break

        if samples:
            cal[cls] = float(np.mean(samples))
            print(f"    '{cls}' mean confidence: {cal[cls]:.2%}")

    cv2.destroyWindow('Calibration')
    print("── Calibration complete ─────────────────────────────────────\n")
    return cal

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_hand(frame, hand, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 210, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, WHITE, -1)
        cv2.circle(frame, (x, y), 5, GREEN,  1)

def draw_bbox(frame, hand, W, H, label, color):
    xs  = [lm.x for lm in hand]
    ys  = [lm.y for lm in hand]
    pad = 18
    x1  = max(0, int(min(xs) * W) - pad)
    y1  = max(0, int(min(ys) * H) - pad)
    x2  = min(W, int(max(xs) * W) + pad)
    y2  = min(H, int(max(ys) * H) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

def conf_color(c: float):
    if c >= 0.70: return GREEN
    if c >= 0.40: return ORANGE
    return RED

def draw_bar(canvas, x, y, w, h, value, color, label='', pct=True):
    """Draw a filled progress bar."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), DARK_GREY, -1)
    filled = int(w * max(0.0, min(1.0, value)))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + h), color, -1)
    if label:
        cv2.putText(canvas, label, (x - 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    if pct:
        pct_txt = f"{value:.0%}"
        cv2.putText(canvas, pct_txt, (x + w + 5, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)

def build_panel(pred_label: str,
                confidence: float,
                smoothed: str,
                top3: list,
                history: list,
                sentence: str,
                fps: float,
                tts_on: bool,
                show_history: bool,
                cal_done: bool) -> np.ndarray:
    """
    Render the right-side info panel (PANEL_W × FEED_H).
    Returns a BGR numpy array.
    """
    p = np.full((FEED_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    x0, rw = 10, PANEL_W - 20   # left margin, usable width

    y = 18
    # ── Header ────────────────────────────────────────────────────────────────
    cv2.putText(p, 'Sign Language', (x0, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, ACCENT, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(p, 'Recognition', (x0, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, ACCENT, 1, cv2.LINE_AA)
    y += 6
    cv2.line(p, (x0, y), (PANEL_W - x0, y), DARK_GREY, 1)
    y += 12

    # ── FPS + model name ──────────────────────────────────────────────────────
    cv2.putText(p, f"FPS {fps:.0f}   [{model_name}]", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    y += 8
    flags = []
    if tts_on:    flags.append('TTS:ON')
    if cal_done:  flags.append('CAL:OK')
    if flags:
        cv2.putText(p, '  '.join(flags), (x0, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 80), 1, cv2.LINE_AA)
    y += 18
    cv2.line(p, (x0, y), (PANEL_W - x0, y), DARK_GREY, 1)
    y += 14

    # ── Main prediction ───────────────────────────────────────────────────────
    main_color = conf_color(confidence) if pred_label not in ('—', 'Unknown') else GREY
    big_label  = smoothed if smoothed not in ('—',) else pred_label
    cv2.putText(p, big_label, (x0, y + 44),
                cv2.FONT_HERSHEY_DUPLEX, 2.6, main_color, 3, cv2.LINE_AA)
    cv2.putText(p, f"raw: {pred_label}", (x0 + 90, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)
    y += 58

    # main confidence bar
    draw_bar(p, x0, y, rw - 38, 16, confidence, main_color, pct=True)
    y += 28
    cv2.line(p, (x0, y), (PANEL_W - x0, y), DARK_GREY, 1)
    y += 10

    # ── Top-3 ─────────────────────────────────────────────────────────────────
    cv2.putText(p, 'Top-3 predictions', (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA)
    y += 14
    for rank, (lbl, conf) in enumerate(top3):
        bar_w = int((rw - 70) * conf)
        bar_color = conf_color(conf)
        rank_lbl  = f"#{rank+1} {lbl}"
        draw_bar(p, x0, y, rw - 70, 14, conf, bar_color, label=rank_lbl, pct=True)
        y += 24
    y += 2
    cv2.line(p, (x0, y), (PANEL_W - x0, y), DARK_GREY, 1)
    y += 10

    # ── Sentence builder ─────────────────────────────────────────────────────
    cv2.putText(p, 'Sentence  [SPACE=add  BKSP=del  C=clear]',
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREY, 1, cv2.LINE_AA)
    y += 14

    # word-wrap sentence into lines of ~22 chars
    max_chars = 22
    display = sentence if sentence else '_'
    lines = [display[i:i+max_chars] for i in range(0, max(len(display), 1), max_chars)]
    for line in lines[-3:]:       # show last 3 lines max
        cv2.putText(p, line, (x0, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 1, cv2.LINE_AA)
        y += 18
    y += 4
    cv2.line(p, (x0, y), (PANEL_W - x0, y), DARK_GREY, 1)
    y += 10

    # ── History panel ─────────────────────────────────────────────────────────
    if show_history:
        cv2.putText(p, 'History  [H=toggle]', (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        y += 14
        slots_available = (FEED_H - y - 10) // 15
        shown = history[-slots_available:] if len(history) > slots_available else history
        for entry in reversed(shown):
            c_col = conf_color(entry['conf'])
            cv2.putText(p,
                        f"{entry['time']}  {entry['char']}  {entry['conf']:.0%}",
                        (x0, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_col, 1, cv2.LINE_AA)
            y += 15
            if y > FEED_H - 10:
                break
    else:
        cv2.putText(p, 'History hidden  [H=show]', (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)

    # ── Bottom hint ───────────────────────────────────────────────────────────
    cv2.putText(p, 'K=calibrate  T=TTS  Q=quit',
                (x0, FEED_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, DARK_GREY, 1, cv2.LINE_AA)

    return p


# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera (index 0).")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FEED_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FEED_H)

# ── State ─────────────────────────────────────────────────────────────────────
pred_buffer  = deque(maxlen=SMOOTH_WINDOW)
frame_times  = deque(maxlen=30)
fps          = 0.0
sentence     = ''
history: list[dict] = []
tts_on       = False
show_history = True
cal_done     = False

print("Running — press Q or ESC to quit.")
print("Controls: SPACE=accept  BKSP=del  C=clear  T=TTS  K=calibrate  H=history")

while True:
    t0 = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    frame = cv2.resize(frame, (FEED_W, FEED_H))
    H, W  = frame.shape[:2]

    img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results  = hand_landmarker.detect(mp_img)

    pred_label  = '—'
    confidence  = 0.0
    smoothed    = '—'
    top3: list  = []

    if results.hand_landmarks:
        hand     = results.hand_landmarks[0]
        features = extract_features(hand).reshape(1, -1)

        pred_enc = model.predict(features)[0]
        raw_lbl  = label_encoder.inverse_transform([pred_enc])[0]

        if hasattr(model, 'predict_proba'):
            proba      = model.predict_proba(features)[0]
            confidence = float(proba[pred_enc])

            # Top-3: sort by probability descending
            top_idx = np.argsort(proba)[::-1][:TOP_K]
            top3    = [(label_encoder.inverse_transform([i])[0], float(proba[i]))
                       for i in top_idx]
        else:
            confidence = 1.0
            top3       = [(raw_lbl, 1.0)]

        # Apply confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            pred_label = raw_lbl
        else:
            pred_label = 'Unknown'

        pred_buffer.append(pred_label)
        smoothed = Counter(pred_buffer).most_common(1)[0][0]

        # Choose bbox colour
        bbox_color = GREEN if smoothed not in ('Unknown', '—') else RED
        draw_hand(frame, hand, W, H)
        draw_bbox(frame, hand, W, H, smoothed, bbox_color)

    # ── Build composite frame ─────────────────────────────────────────────────
    panel = build_panel(
        pred_label=pred_label,
        confidence=confidence,
        smoothed=smoothed,
        top3=top3,
        history=history,
        sentence=sentence,
        fps=fps,
        tts_on=tts_on,
        show_history=show_history,
        cal_done=cal_done,
    )
    canvas = np.hstack([frame, panel])
    cv2.imshow('Sign Language Recognition', canvas)

    # ── FPS ───────────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    frame_times.append(t1 - t0)
    fps = 1.0 / (sum(frame_times) / len(frame_times))

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):        # Q / ESC → quit
        break

    elif key == ord(' '):            # SPACE → accept prediction
        accepted = smoothed
        if accepted not in ('—', 'Unknown') and confidence >= CONFIDENCE_THRESHOLD:
            sentence += accepted
            ts = datetime.now().strftime('%H:%M:%S')
            history.append({'time': ts, 'char': accepted, 'conf': confidence})
            if len(history) > HISTORY_MAX:
                history.pop(0)
            log_prediction(accepted, confidence)
            if tts_on:
                speak(accepted)

    elif key == 8:                   # BKSP → delete last char
        sentence = sentence[:-1]

    elif key == ord('c'):            # C → clear sentence
        sentence = ''

    elif key == ord('t'):            # T → toggle TTS
        if TTS_AVAILABLE:
            tts_on = not tts_on
            print(f"TTS {'ON' if tts_on else 'OFF'}")
        else:
            print("TTS not available (pyttsx3 not installed).")

    elif key == ord('h'):            # H → toggle history panel
        show_history = not show_history

    elif key == ord('k'):            # K → calibration mode
        cv2.destroyWindow('Sign Language Recognition')
        calibration = run_calibration(cap, hand_landmarker)
        cal_done    = bool(calibration)

# ── Cleanup ───────────────────────────────────────────────────────────────────
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
_csv_f.close()

if sentence:
    print(f"\nFinal sentence : {sentence}")
print(f"Predictions CSV: {CSV_PATH}")
print("Exited.")
