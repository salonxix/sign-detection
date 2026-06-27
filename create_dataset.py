"""
create_dataset.py
-----------------
Processes all images in ./data/, extracts MediaPipe hand landmarks,
applies preprocessing (centering, scaling, normalization), and saves
the feature vectors to data.pickle.

Preprocessing pipeline (per hand):
  1. Extract 21 (x, y) landmark coordinates  → raw 42-dim vector
  2. Translate  : subtract wrist (landmark 0) to center the hand
  3. Scale      : divide by the max absolute coordinate so all values ∈ [-1, 1]
  4. Normalize  : L2-normalize the full feature vector to unit length

Usage:
  python create_dataset.py
"""

import os
import pickle

import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── MediaPipe HandLandmarker setup ────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"hand_landmarker.task not found at {MODEL_PATH}.\n"
        "Download it with:\n"
        "  python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')\""
    )

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ── Preprocessing helper ──────────────────────────────────────────────────────
def extract_features(hand_landmarks) -> np.ndarray:
    """
    Given a list of 21 NormalizedLandmark objects, returns a preprocessed
    feature vector of length 42 (21 landmarks × 2 coordinates).

    Steps:
      1. Build raw (x, y) pairs.
      2. Translate so wrist (index 0) is at the origin.
      3. Scale so the bounding box fits in [-1, 1] (scale-invariant).
      4. L2-normalize the full vector (pose-direction invariant).
    """
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float32)  # (21, 2)

    # Step 2 – translate: center on wrist
    coords -= coords[0]

    # Step 3 – scale: divide by max absolute value so coords ∈ [-1, 1]
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords /= max_abs

    # Step 4 – L2-normalize the flattened vector
    flat = coords.flatten()                          # (42,)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat /= norm

    return flat


# ── Dataset extraction ────────────────────────────────────────────────────────
DATA_DIR = './data'

data    = []
labels  = []
processed = 0
skipped   = 0

all_dirs = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Found {len(all_dirs)} class folders: {all_dirs}")

for dir_ in all_dirs:
    dir_path   = os.path.join(DATA_DIR, dir_)
    img_files  = [f for f in os.listdir(dir_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    class_ok = 0
    for img_name in img_files:
        full_path = os.path.join(dir_path, img_name)
        img = cv2.imread(full_path)
        if img is None:
            skipped += 1
            continue

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = hand_landmarker.detect(mp_image)

        if results.hand_landmarks:
            features = extract_features(results.hand_landmarks[0])
            data.append(features)
            labels.append(dir_)
            class_ok  += 1
            processed += 1
        else:
            skipped += 1

    print(f"  Class '{dir_}': {class_ok}/{len(img_files)} images used")

hand_landmarker.close()

print(f"\nTotal processed : {processed}")
print(f"Total skipped   : {skipped}  (unreadable or no hand detected)")

if processed == 0:
    raise RuntimeError("No samples extracted — check your data/ folder and images.")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'data'   : np.array(data,   dtype=np.float32),
    'labels' : np.array(labels, dtype=str),
}
with open('data.pickle', 'wb') as f:
    pickle.dump(output, f)

print("data.pickle saved  →  shape:", output['data'].shape)
