# Real-Time Sign Language Recognition System

Recognises **36 classes** — digits **0–9** and letters **A–Z** — from a live webcam feed using MediaPipe hand landmarks and an auto-selected ML classifier.

---

## Project Structure

```
├── collect_imgs.py          # Step 1 – capture training images
├── create_dataset.py        # Step 2 – extract landmarks → data.pickle
├── train_classifier.py      # Step 3 – train + evaluate → model.p
├── inference_classifier.py  # Step 4 – real-time recognition UI
│
├── hand_landmarker.task     # MediaPipe model bundle (download once)
├── data/                    # Training images (36 sub-folders)
├── metrics/
│   ├── confusion_matrix.png # Saved after every training run
│   └── report.json          # Per-model accuracy + per-class F1
│
├── data.pickle              # Preprocessed feature matrix
├── old_data.pickle          # Snapshot used for incremental retraining
├── model.p                  # Best trained model bundle
├── label_map.json           # Integer → class name mapping
└── predictions.csv          # Live prediction log (appended during inference)
```

---

## Requirements

- Python **3.12** (3.10+ works)
- Webcam

### Install

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| opencv-python | 4.10.0.84 |
| mediapipe | 0.10.35 |
| scikit-learn | 1.5.2 |
| numpy | < 2 |
| xgboost | 2.1.4 |
| pyttsx3 | 2.98 |
| seaborn | 0.13.2 |
| matplotlib | 3.11.0 |

### Download MediaPipe hand model (one-time)

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task',
    'hand_landmarker.task')
print('Done')
"
```

---

## Training Pipeline

### Step 1 – Collect images

```bash
python collect_imgs.py
```

- Iterates all 36 classes (0–9 then A–Z).
- Captures **500 images per class** (18 000 total).
- Re-runs resume from where you left off.

| Key | Action |
|---|---|
| Q | Start collecting for current class |
| S | Skip current class |
| ESC | Abort session |

### Step 2 – Build dataset

```bash
python create_dataset.py
```

Runs MediaPipe on every image and applies the preprocessing pipeline:

| Step | Operation | Why |
|---|---|---|
| Translate | Subtract wrist (landmark 0) | Position-independent |
| Scale | Divide by max abs value → [−1, 1] | Scale-independent |
| L2-normalize | Divide by Euclidean norm | Orientation-invariant |

Output: `data.pickle` — shape `(N, 42)`.

### Step 3 – Train classifiers

```bash
# Normal training
python train_classifier.py

# Retrain after collecting more data — merges with old_data.pickle
python train_classifier.py --incremental

# Force save even if new model is worse than the saved one
python train_classifier.py --incremental --force
```

Three models are trained and compared:

| Classifier | Config |
|---|---|
| Random Forest | 200 trees |
| SVM RBF | C=10, gamma=scale, probability=True |
| XGBoost | 300 estimators, depth 6 |

Each is evaluated with **5-fold CV + held-out 20% test set**.
The best-accuracy model is automatically saved to `model.p`.

**Outputs after training:**

| File | Contents |
|---|---|
| `model.p` | Best model + LabelEncoder + class list |
| `label_map.json` | `{"0": "0", "1": "1", … "35": "Z"}` |
| `metrics/confusion_matrix.png` | Heatmap for the best model |
| `metrics/report.json` | Per-model CV acc, test acc, per-class precision/recall/F1 |
| `old_data.pickle` | Snapshot of current data for next incremental run |

The trainer **never downgrades**: if the new model scores lower than the saved one, `model.p` is left untouched (override with `--force`).

### Step 4 – Run inference

```bash
python inference_classifier.py
```

---

## Inference Controls

| Key | Action |
|---|---|
| SPACE | Accept current prediction → append to sentence |
| BKSP | Delete last character from sentence |
| C | Clear sentence |
| T | Toggle text-to-speech on/off |
| K | Enter calibration mode |
| H | Toggle history panel |
| Q / ESC | Quit |

---

## Inference Features

### Confidence threshold
Predictions below **70%** confidence are shown as **"Unknown"** and cannot be added to the sentence. Adjust `CONFIDENCE_THRESHOLD` at the top of `inference_classifier.py`.

### Top-3 predictions
The right panel shows the top 3 candidate classes with colour-coded confidence bars:

| Colour | Confidence |
|---|---|
| Green | ≥ 70% |
| Orange | 40–69% |
| Red | < 40% |

### Smoothing
Majority vote over the **last 10 frames** suppresses single-frame flicker. The smoothed label drives the bounding box and the sentence builder.

### Sentence builder
Press SPACE whenever the smoothed label is stable to append a character. Build words and phrases in the on-screen sentence area.

### Text-to-speech
Press T to toggle. Each accepted character is spoken aloud via `pyttsx3` (Windows SAPI / macOS NSSpeechSynthesizer / Linux espeak). Runs in a background thread so it never blocks the camera feed.

### Calibration mode
Press K before starting recognition. For each class the system asks you to hold the sign for 60 frames and measures the typical model confidence. This baseline is stored in memory and can be used to spot systematic under-confidence for specific signs. Re-run whenever lighting or hand position changes significantly.

### Prediction history
The right panel shows the last 20 accepted predictions with timestamps and confidence. Toggle with H.

### CSV export
Every accepted prediction is appended to `predictions.csv`:

```
timestamp,character,confidence,model
2025-01-15T14:32:01,A,0.9412,SVM_RBF
2025-01-15T14:32:03,B,0.8871,SVM_RBF
```

---

## Incremental Retraining

To add new classes or more data without losing existing samples:

1. Collect new images: `python collect_imgs.py`
2. Re-extract features: `python create_dataset.py`
3. Retrain merging old samples: `python train_classifier.py --incremental`

`old_data.pickle` is updated automatically on every training run so you can chain as many incremental runs as you like.

---

## Tips

- **Lighting** — avoid strong backlighting; soft frontal light is ideal.
- **Background** — plain backgrounds help MediaPipe detect hands reliably.
- **Variety** — slightly vary angle and distance when collecting to improve generalisation.
- **Balance** — keep roughly equal image counts per class (500 is a good target).
- **Low confidence?** — run calibration mode (K) and consider collecting more images for the problem classes.
