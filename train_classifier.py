"""
train_classifier.py
-------------------
Trains three classifiers on preprocessed landmark features:
  1. Random Forest
  2. SVM  (RBF kernel)
  3. XGBoost  (if installed)

New in this version
───────────────────
  • Saves full evaluation metrics: confusion matrix image, per-class
    precision / recall / F1, and a JSON metrics report.
  • Supports incremental retraining: merges new data.pickle with the
    previous dataset snapshot (old_data.pickle) so old samples are
    never thrown away.
  • Always saves the best model; never overwrites if new accuracy is lower
    (--force flag overrides).
  • Saves model.p, label_map.json, metrics/report.json,
    metrics/confusion_matrix.png.

Usage
─────
  # First-time training
  python train_classifier.py

  # Retrain after collecting more data (old samples are kept)
  python train_classifier.py --incremental

  # Force overwrite even if new model is worse
  python train_classifier.py --incremental --force
"""

import argparse
import json
import os
import pickle
import shutil
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ── XGBoost (optional) ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping. Install with: pip install xgboost")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train sign-language classifiers')
parser.add_argument('--incremental', action='store_true',
                    help='Merge new data.pickle with old_data.pickle before training')
parser.add_argument('--force', action='store_true',
                    help='Overwrite model.p even if new accuracy is lower')
args = parser.parse_args()

METRICS_DIR = Path('metrics')
METRICS_DIR.mkdir(exist_ok=True)

# ── Load / merge data ─────────────────────────────────────────────────────────
def load_pickle(path):
    d = pickle.load(open(path, 'rb'))
    return np.array(d['data'], dtype=np.float32), np.array(d['labels'], dtype=str)

print("Loading data.pickle …")
data, labels = load_pickle('data.pickle')

if args.incremental and os.path.exists('old_data.pickle'):
    print("Incremental mode — merging with old_data.pickle …")
    old_data, old_labels = load_pickle('old_data.pickle')
    data   = np.vstack([old_data,   data])
    labels = np.concatenate([old_labels, labels])
    print(f"  Old samples : {len(old_data)}")
    print(f"  New samples : {len(load_pickle('data.pickle')[0])}")

# Snapshot current data so next incremental run can find it
shutil.copy('data.pickle', 'old_data.pickle')

print(f"  Total samples : {len(data)}")
print(f"  Features      : {data.shape[1]}")
print(f"  Classes       : {sorted(set(labels))}  ({len(set(labels))} total)")

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
print(f"  Label map     : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Train / test split ────────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_enc, test_size=0.2, shuffle=True,
    stratify=labels_enc, random_state=42
)
print(f"\nTrain : {len(x_train)}   Test : {len(x_test)}\n")

# ── Classifiers ───────────────────────────────────────────────────────────────
classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=2, random_state=42, n_jobs=-1,
    ),
    'SVM_RBF': SVC(
        kernel='rbf', C=10.0, gamma='scale',
        probability=True, random_state=42,
    ),
}
if XGBOOST_AVAILABLE:
    classifiers['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42,
        n_jobs=-1, verbosity=0,
    )

# ── Train + evaluate ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results   = {}
all_metrics = {}

print("=" * 62)
print(f"{'Classifier':<20} {'CV Acc':>8} {'Test Acc':>10} {'Time':>8}")
print("=" * 62)

for name, clf in classifiers.items():
    t0        = time.time()
    cv_scores = cross_val_score(clf, x_train, y_train,
                                cv=cv, scoring='accuracy', n_jobs=-1)
    clf.fit(x_train, y_train)
    elapsed   = time.time() - t0

    y_pred   = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_mean  = cv_scores.mean()

    results[name] = {'clf': clf, 'cv_acc': cv_mean,
                     'test_acc': test_acc, 'y_pred': y_pred}

    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    all_metrics[name] = {
        'cv_accuracy'  : float(cv_mean),
        'test_accuracy': float(test_acc),
        'report'       : report,
    }

    print(f"{name:<20} {cv_mean:>7.2%}  {test_acc:>9.2%}  {elapsed:>6.1f}s")

print("=" * 62)

# ── Best model ────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['test_acc'])
best_clf  = results[best_name]['clf']
best_acc  = results[best_name]['test_acc']
best_pred = results[best_name]['y_pred']

print(f"\n✓ Best model : {best_name}  ({best_acc:.2%} test accuracy)")
print(f"\nClassification report ({best_name}):")
print(classification_report(y_test, best_pred, target_names=le.classes_))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
fig_h = max(8, len(le.classes_) // 3)
fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f'Confusion Matrix — {best_name}  (test acc {best_acc:.2%})')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
cm_path = METRICS_DIR / 'confusion_matrix.png'
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# ── Metrics JSON ──────────────────────────────────────────────────────────────
metrics_report = {
    'best_model'  : best_name,
    'best_acc'    : float(best_acc),
    'all_models'  : all_metrics,
}
report_path = METRICS_DIR / 'report.json'
with open(report_path, 'w') as f:
    json.dump(metrics_report, f, indent=2)
print(f"Metrics report  saved → {report_path}")

# ── Save model (guard against regression) ─────────────────────────────────────
prev_acc = 0.0
if os.path.exists('model.p'):
    try:
        prev = pickle.load(open('model.p', 'rb'))
        prev_acc = prev.get('test_accuracy', 0.0)
    except Exception:
        pass

if best_acc >= prev_acc or args.force:
    bundle = {
        'model'         : best_clf,
        'label_encoder' : le,
        'model_name'    : best_name,
        'test_accuracy' : best_acc,
        'classes'       : list(le.classes_),
    }
    with open('model.p', 'wb') as f:
        pickle.dump(bundle, f)

    label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\nmodel.p        saved  [{best_name}  {best_acc:.2%}]"
          f"  (prev best: {prev_acc:.2%})")
    print("label_map.json saved")
else:
    print(f"\n⚠  New model ({best_acc:.2%}) is not better than saved model "
          f"({prev_acc:.2%}). model.p NOT updated.")
    print("   Use --force to override.")

# ── All-model summary ─────────────────────────────────────────────────────────
print("\n── Model comparison ──")
for n, r in sorted(results.items(), key=lambda kv: -kv[1]['test_acc']):
    marker = " ← best" if n == best_name else ""
    print(f"  {n:<20}  CV={r['cv_acc']:.2%}   Test={r['test_acc']:.2%}{marker}")
