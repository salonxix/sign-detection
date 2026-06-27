"""
collect_imgs.py
---------------
Captures training images from your webcam for all 36 classes:
  - Digits  0-9  (folders named '0' .. '9')
  - Letters A-Z  (folders named 'A' .. 'Z')

Usage:
  python collect_imgs.py

Controls:
  Press Q  - start collecting images for the current class
  Press S  - skip the current class (keeps existing images)
  Press ESC- abort the entire collection session
"""

import os
import cv2

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR      = './data'
DATASET_SIZE  = 500          # images per class
CLASSES       = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
# CLASSES = ['0','1',...,'9','A','B',...,'Z']  → 36 total

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

print(f"Data directory  : {os.path.abspath(DATA_DIR)}")
print(f"Classes         : {CLASSES}")
print(f"Images per class: {DATASET_SIZE}")
print(f"Total images    : {len(CLASSES) * DATASET_SIZE}")
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera (index 0). Check your webcam.")

for j, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    existing  = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])

    print(f"[{j+1}/{len(CLASSES)}] Class '{class_name}'  "
          f"({existing} images already collected)")

    # ── Wait-for-ready screen ─────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame,
                    f"Class: {class_name}  ({existing}/{DATASET_SIZE} exist)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    "Press Q=Collect  S=Skip  ESC=Abort",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow('Collect Images', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"  → Skipped '{class_name}'")
            break
        elif key == 27:          # ESC
            print("Aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

    # ── Collect images ────────────────────────────────────────────────────
    counter = existing          # continue from where we left off
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Progress bar
        progress = int((counter / DATASET_SIZE) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0] - 12),
                      (progress, frame.shape[0]), (0, 200, 0), -1)
        cv2.putText(frame,
                    f"Collecting '{class_name}'  {counter}/{DATASET_SIZE}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collect Images', frame)
        cv2.waitKey(1)

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f"  → Collected {counter} images for class '{class_name}'")

cap.release()
cv2.destroyAllWindows()
print("\nCollection complete.")
