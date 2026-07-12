"""
train_model.py
----------------
Trains a neural network to classify hand-landmark snapshots into
sentence-gesture labels, using the data.csv produced by collect_data.py.

WHAT'S NEW (accuracy overhaul)
- Features are NORMALIZED (wrist-centered + scale-normalized) via
  hand_features.normalize(), so the model learns hand SHAPE, not screen
  position or camera distance. This is the single biggest accuracy fix.
- The TRAINING split is AUGMENTED (rotation/scale/jitter/translate + mirror)
  via hand_features.augment() to generalize far better from limited data.
  The TEST split is only normalized (never augmented) for an honest score.
- Class weights counter label imbalance.
- Prints a confusion matrix + per-class precision/recall so you can see which
  signs get confused and target re-collection.

HOW TO USE
1. Collect data for >= 2 gestures (run collect_data.py first).
2. Run: python train_model.py
3. It prints test accuracy + a confusion matrix, then saves model.h5 and
   labels.json.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

import hand_features as hf

DATA_FILE = "data.csv"
MODEL_FILE = "model.h5"
LABELS_FILE = "labels.json"

N_AUG = 6            # random variants per sample (x2 with mirror) for training
RANDOM_STATE = 42


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(hf.FEATURE_DIM,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def normalize_rows(raw_rows):
    """Normalize a batch of raw 126-vectors (no augmentation) -> (N, 126)."""
    return np.array([hf.normalize(r) for r in raw_rows], dtype="float32")


def augment_rows(raw_rows, labels):
    """Augment + normalize training rows. Returns (features, expanded_labels)."""
    feats, ys = [], []
    for raw, y in zip(raw_rows, labels):
        variants = hf.augment(raw, n_aug=N_AUG, include_mirror=True)
        feats.append(variants)
        ys.extend([y] * len(variants))
    return np.concatenate(feats, axis=0).astype("float32"), np.array(ys)


def main():
    df = pd.read_csv(DATA_FILE)

    expected = hf.csv_header()
    if list(df.columns) != expected:
        print("ERROR: data.csv has an unexpected format "
              f"({df.shape[1]} cols). Expected {len(expected)} "
              "(126 landmark cols + label).")
        print("Re-collect with the updated collect_data.py first.")
        return

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} gestures:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        print("\nYou need at least 2 different gestures to train a classifier.")
        return

    X_raw = df.drop(columns=["label"]).values.astype("float32")
    y_text = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    num_classes = len(encoder.classes_)

    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Augment ONLY the training split; test is normalized-only for an honest score.
    X_train, y_train_aug = augment_rows(X_train_raw, y_train)
    X_test = normalize_rows(X_test_raw)
    print(f"\nTraining features after augmentation: {X_train.shape[0]} "
          f"(from {X_train_raw.shape[0]} raw samples)")

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_aug), y=y_train_aug
    )
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train_aug), class_weights)}

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    model.fit(
        X_train, y_train_aug,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc * 100:.1f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nPer-class report:")
    print(classification_report(
        y_test, y_pred, target_names=list(encoder.classes_), zero_division=0
    ))

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_test, y_pred)
    labels = list(encoder.classes_)
    width = max(len(l) for l in labels) + 1
    header = " " * width + "".join(f"{i:>4}" for i in range(len(labels)))
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:<{width}}" + "".join(f"{v:>4}" for v in row) + f"  ({i})")
    print("\nBig off-diagonal numbers = confused sign pairs; collect more, "
          "more-distinct samples for those.")

    model.save(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Labels saved to {LABELS_FILE}")


if __name__ == "__main__":
    main()
