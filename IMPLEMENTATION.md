# Implementation Plan — Accuracy Overhaul

This document tracks the staged rebuild of the sign-language recognizer. The
goal is to fix the app's poor real-world accuracy and add a path to motion
signs.

## Why the old app was inaccurate

The model was fed MediaPipe's **raw image-space `x,y` coordinates** (84 numbers).
That teaches the model *where the hand is in the frame* and *how large it looks*
(distance to camera) instead of the hand's **shape**. Any change of position or
camera distance from the collection setup breaks it. Held-out test accuracy
looked fine only because the split shared the same framing bias.

Contributing factors: depth (`z`) discarded; handedness-slot assignment fragile
under the mirror flip; several signs are inherently *motion* (Hello, Thank you,
No, How, Yes) and cannot be captured by one static frame; no augmentation; class
imbalance ignored; feature-extraction code duplicated across four files
(train/serve skew).

---

## Stage 1 — Fix the static pipeline (biggest win)

### `hand_features.py` (new, shared by everything)
- `make_hands()` — single MediaPipe `Hands` config.
- `extract_raw_hands(results)` — per-hand raw landmarks **with z**: 21×3 per
  hand, Left slot then Right slot, missing hand = zeros → **126 numbers**.
- `normalize(raw_vec)` — the core fix, per hand independently:
  1. translate: subtract wrist (landmark 0) → wrist at origin (position
     invariance);
  2. scale: divide by wrist→middle-MCP distance (landmark 0→9) (distance/scale
     invariance).
  Rotation is left intact (orientation is meaningful in ASL). Missing hand
  stays zeros.
- `augment(raw_vec)` — training only: jitter, scale, translate, and a **mirror**
  variant (swap L/R + negate x) for robustness to handedness flips.
- Constants: `RAW_DIM = 126`, `PER_HAND = 63`.

### `collect_data.py`
- Import from `hand_features`; drop the local copy.
- CSV header → 126 cols (`{L,R}{0..20}_{x,y,z}`) + `label` = 127.
- Continuous-capture mode (hold key → ~5 samples/sec) for fast, varied data.
- Archive any existing 84-col `data.csv` (schema change).

### `train_model.py`
- Load raw 126-dim → `normalize` → `augment` to expand training set.
- Class weights (`compute_class_weight`) for imbalance.
- Model: `Input(126) → Dense(256)+BatchNorm+Dropout → Dense(128)+Dropout →
  Dense(64) → softmax`; `EarlyStopping` + `ReduceLROnPlateau`.
- Print **confusion matrix + classification_report** to expose colliding signs.
- Save `model.h5` + `labels.json` (unchanged filenames).

### `app.py` / `live_translate.py`
- Use `extract_raw_hands` + `normalize` from the shared module.
- `LandmarkSmoother` smooths the raw 126-dim vector (63/hand); `normalize`
  applied per frame before `predict`. Threshold + majority-vote unchanged.

---

## Stage 2 — Motion / sequence model (follow-on)

- `collect_sequences.py` — record ~30-frame clips of raw landmarks per sample.
- `train_sequence_model.py` — normalize each frame → GRU/LSTM over `[T,126]` →
  softmax. A held pose is a constant sequence, so this can subsume the static
  model.
- Inference: sliding buffer of last T frames in the apps; reuse smoothing.
- Keep static + sequence selectable by flag until the sequence model wins.

---

## Verification
1. Import sanity check.
2. Collect ~50 samples for a few signs (continuous mode); confirm 127 columns.
3. Train; read held-out accuracy **and** confusion matrix.
4. **Real-world test:** run `app.py`, sign while moving around the frame and
   changing camera distance — the exact case normalization fixes.
5. Desktop parity: `live_translate.py` matches the web app.
6. Stage 2: a motion sign recognized where the static model was ambiguous.

## Status
- [x] IMPLEMENTATION.md
- [x] hand_features.py — normalization verified translation/scale-invariant
- [x] collect_data.py — 126-col CSV + burst capture + legacy archive
- [x] train_model.py — normalize+augment+class weights+confusion matrix (pipeline smoke-tested)
- [x] app.py — shared core, per-frame normalize, real hands count
- [x] live_translate.py — shared core (train/serve parity)
- [x] Stage 2: collect_sequences.py + train_sequence_model.py + live_sequence.py (GRU, smoke-tested)

## How to run
Stage 1 (static): `python collect_data.py` → `python train_model.py` → `python app.py` (or `live_translate.py`)
Stage 2 (motion): `python collect_sequences.py` → `python train_sequence_model.py` → `python live_sequence.py`

NOTE: the CSV schema changed (84 → 126 cols to add depth z), so the old
`data.csv` cannot be reused — collect_data.py auto-archives it to
`data_legacy_xyonly.csv` and you must re-collect before training.

## Open-set rejection (untrained gestures)

A softmax classifier is closed-world: it always distributes 100% confidence
across the *trained* signs, so an untrained gesture is mapped to the nearest
known sign instead of "unknown." Mitigations in place:

- **Two-part gate** in `app.py` / `live_translate.py`: a prediction is only shown
  if the top class clears `CONFIDENCE_THRESHOLD` (0.85) **and** beats the
  runner-up by `MARGIN_THRESHOLD` (0.30). Ambiguous frames read as nothing.
- **Recommended `NONE` class**: collect a background/non-sign label in
  `collect_data.py` and retrain so unknown input has an explicit bucket (details
  in `SETUP_NOTES.md`, Step 6). This is the most effective fix.
