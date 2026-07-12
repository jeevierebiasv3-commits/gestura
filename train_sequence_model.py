"""
train_sequence_model.py  (Stage 2 — motion signs)
-------------------------------------------------
Trains a GRU sequence model on the motion clips recorded by
collect_sequences.py, so the app can recognize MOTION-based signs.

Pipeline per clip:
  raw (T, 126) -> resample to SEQ_LEN -> normalize each frame -> (augment) ->
  GRU -> softmax over sign labels.

Outputs: seq_model.h5 + seq_labels.json  (used by live_sequence.py).

HOW TO USE
1. Record clips with collect_sequences.py (>= 2 labels, ~20-40 clips each).
2. Run: python train_sequence_model.py
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

import hand_features as hf

SEQ_DIR = "sequences"
MODEL_FILE = "seq_model.h5"
LABELS_FILE = "seq_labels.json"
SEQ_LEN = hf.SEQ_LEN
N_AUG = 4
RANDOM_STATE = 42

# --- dataset-health thresholds (per class) ---
CLIPS_GOOD = 30        # >= this is a healthy class
CLIPS_OK = 15          # >= this trains, but more is better
CLIPS_MIN = 5          # below this the class is barely learnable
IMBALANCE_RATIO = 3.0  # warn if largest class is > this x the smallest
# labels that count as an explicit "not a sign" bucket (open-set rejection)
NONE_HINTS = {"none", "idle", "nothing", "background", "rest",
              "not a hand sign", "this isn't a hand sign", "unknown"}


def dataset_health(labels, classes):
    """Print a per-class breakdown and flag problems that hurt accuracy.

    Returns (ok_to_train, drop_classes) where drop_classes are labels with too
    few clips to appear in both a train and a test split (they would crash the
    stratified split or teach the model almost nothing)."""
    counts = {c: labels.count(c) for c in classes}
    total = sum(counts.values())
    biggest = max(counts.values())
    smallest = min(counts.values())
    bar_unit = max(1, biggest // 30)   # scale bars so the largest is ~30 chars

    print("\n" + "=" * 58)
    print("DATASET HEALTH")
    print("=" * 58)
    print(f"{total} clips across {len(classes)} signs\n")

    warnings, criticals, drop = [], [], []
    for c in classes:
        n = counts[c]
        bar = "#" * (n // bar_unit)
        if n < 2:
            tag = "CANNOT TRAIN (needs >= 2)"
            criticals.append(c)
            drop.append(c)
        elif n < CLIPS_MIN:
            tag = "CRITICALLY LOW"
            criticals.append(c)
        elif n < CLIPS_OK:
            tag = "low - add more"
            warnings.append(c)
        elif n < CLIPS_GOOD:
            tag = "ok"
        else:
            tag = "good"
        print(f"  {c:<28} {n:>4}  {bar:<32} {tag}")

    # imbalance check
    imbalanced = smallest > 0 and (biggest / smallest) > IMBALANCE_RATIO

    # NONE / background class present?
    has_none = any(c.strip().lower() in NONE_HINTS for c in classes)

    print("\n" + "-" * 58)
    if criticals:
        print(f"! {len(criticals)} class(es) with too few clips: "
              f"{', '.join(criticals)}")
        print(f"    Aim for >= {CLIPS_GOOD} clips each (>= {CLIPS_OK} minimum). "
              f"Record more with collect_sequences.py.")
    if warnings:
        print(f"~ {len(warnings)} class(es) a bit thin: {', '.join(warnings)} "
              f"(target >= {CLIPS_GOOD}).")
    if imbalanced:
        print(f"~ Imbalanced: largest class is {biggest / smallest:.1f}x the "
              f"smallest. Class weights help, but even counts train better.")
    if not has_none:
        print("~ No NONE / background class found. Without an explicit "
              "'not a sign' bucket, unknown gestures get forced into the")
        print("    nearest real sign (e.g. a peace sign read as 'Hi'). Add a "
              "NONE label of varied non-signs and retrain.")
    if not (criticals or warnings or imbalanced or not has_none):
        print("All classes look healthy.")

    if drop:
        print(f"\n  Skipping (only 1 clip, can't split): {', '.join(drop)}")
    print("=" * 58 + "\n")

    return True, drop


def load_clips():
    """Return (clips list of raw (T,126), labels list of str)."""
    clips, labels = [], []
    if not os.path.isdir(SEQ_DIR):
        return clips, labels
    for label in sorted(os.listdir(SEQ_DIR)):
        d = os.path.join(SEQ_DIR, label)
        if not os.path.isdir(d):
            continue
        # A folder name may be a Windows-sanitized version of the real label
        # (e.g. "How are you_" for "How are you?"). collect_sequences.py writes
        # the true label to label.txt; prefer it so punctuation is preserved.
        true_label = label
        sidecar = os.path.join(d, "label.txt")
        if os.path.isfile(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                true_label = text
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npy"):
                clips.append(np.load(os.path.join(d, fn)).astype("float32"))
                labels.append(true_label)
    return clips, labels


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, hf.SEQ_FEATURE_DIM)),
        layers.Masking(mask_value=0.0),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    clips, labels = load_clips()
    if len(clips) == 0:
        print(f"No clips found under ./{SEQ_DIR}/. Run collect_sequences.py first.")
        return

    classes = sorted(set(labels))
    if len(classes) < 2:
        print("Need at least 2 motion labels to train.")
        return

    # --- dataset-health report (and drop classes too small to split) ---
    _, drop = dataset_health(labels, classes)
    if drop:
        keep = [(c, l) for c, l in zip(clips, labels) if l not in drop]
        clips = [c for c, _ in keep]
        labels = [l for _, l in keep]
        classes = sorted(set(labels))
        if len(classes) < 2:
            print("Not enough trainable classes left after dropping tiny ones.")
            return

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[l] for l in labels])
    with open(LABELS_FILE, "w") as f:
        json.dump({int(i): c for c, i in label_to_idx.items()}, f, indent=2)

    # resample every clip to fixed length up front
    fixed = [hf.resample_sequence(c, SEQ_LEN) for c in clips]

    idx = np.arange(len(fixed))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                      random_state=RANDOM_STATE, stratify=y)

    # train: normalize + augment ; test: normalize only
    Xtr, ytr = [], []
    for i in tr_idx:
        variants = hf.augment_sequence(fixed[i], n_aug=N_AUG, include_mirror=True)
        Xtr.append(variants)
        ytr.extend([y[i]] * len(variants))
    Xtr = np.concatenate(Xtr, axis=0).astype("float32")
    ytr = np.array(ytr)

    Xte = np.stack([hf.seq_features(fixed[i]) for i in te_idx]).astype("float32")
    yte = y[te_idx]

    print(f"\nTrain clips after augmentation: {Xtr.shape[0]}  |  test: {Xte.shape[0]}")

    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(ytr), cw)}

    model = build_model(len(classes))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]

    model.fit(Xtr, ytr, validation_data=(Xte, yte),
              epochs=150, batch_size=16, class_weight=class_weight,
              callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.1f}%")

    # Save FIRST, before the report — a class with no test samples (tiny class,
    # unlucky split) makes classification_report's label/target_names counts
    # mismatch and raise, and we don't want that to throw away a trained model.
    model.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Labels saved to {LABELS_FILE}")

    if len(te_idx) > 0:
        y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
        print("\nPer-class report:")
        # pass explicit labels so the report always lines up with `classes`
        # even when some class is absent from this particular test split
        all_labels = list(range(len(classes)))
        print(classification_report(yte, y_pred, labels=all_labels,
                                    target_names=classes, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(yte, y_pred, labels=all_labels))


if __name__ == "__main__":
    main()
