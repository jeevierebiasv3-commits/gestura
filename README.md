# Sign Sentence Translator

Recognizes whole-sentence hand gestures (both hands) from your laptop's
webcam using MediaPipe for hand tracking and a neural network for
classification — no letter-by-letter spelling involved.

There are two pipelines:

- **Stage 1 — Static poses:** classifies a single held hand shape per frame.
  Good for signs that are a fixed pose.
- **Stage 2 — Motion signs:** classifies short clips with a GRU sequence model,
  so movement-based signs (a wave for "Hello", hand-from-chin for "Thank you",
  a finger-wag for "No") are recognized properly.

**Accuracy note:** landmarks are **normalized** (each hand is re-centered on the
wrist and scaled by hand size) before the model sees them, and include depth
(z). This makes recognition robust to *where* your hand is in the frame and
*how far* it is from the camera — the main cause of earlier misreads. All
scripts share this logic via `hand_features.py`.

## Setup

1. Install Python 3.9–3.11 (TensorFlow doesn't yet support the very latest
   Python versions — check before installing if unsure).
2. Open a terminal in this folder and run:

   ```
   pip install -r requirements.txt
   ```

   This installs OpenCV (webcam access), MediaPipe (hand tracking),
   TensorFlow (the ML model), and a few helper libraries.

## Usage — 3 steps

### Step 1: Collect training data
```
python collect_data.py
```
- Type a label like `I_AM_HUNGRY` (no spaces — use underscores).
- A webcam window opens. Make the gesture pose with your hand(s).
- Press **SPACE** ~30-50 times to capture samples, slightly shifting your
  hand position/angle/distance from camera between captures.
- Press **n** to move to the next gesture and type a new label.
- Press **q** when done with all gestures. This creates `data.csv`.

**Important:** Record at least 2 different gestures so the model has
something to distinguish between. More samples per gesture (30-50+) and
more variation (angle, distance, slight rotation) = better accuracy.

### Step 2: Train the model
```
python train_model.py
```
This reads `data.csv`, trains a neural network, prints test accuracy, and
saves `model.h5` (the trained model) and `labels.json` (label lookup).

If accuracy is low, usually it means: too few samples, gestures too similar
to each other, or not enough variation during collection. Just go back to
Step 1 and add more/better data, then retrain.

### Step 3: Run the live translator
```
python live_translate.py
```
Opens your webcam, tracks your hands, and displays the predicted sentence
as text on screen in real time. Press **q** to quit.

## How it works under the hood

- MediaPipe detects up to 2 hands and gives 21 (x, y) landmark points per
  hand — fingertips, knuckles, wrist, etc.
- We flatten both hands into one list of 84 numbers per frame (a missing
  hand is filled with zeros).
- `collect_data.py` saves these number-lists with a label you choose.
- `train_model.py` trains a small feedforward neural network
  (84 numbers in → gesture label out).
- `live_translate.py` runs that trained model on every webcam frame and
  smooths predictions over several frames so the displayed text doesn't
  flicker between guesses.

## Adding more gestures later

Just re-run `collect_data.py` (it appends to the same `data.csv`), then
re-run `train_model.py` to retrain on the combined data. You don't need to
touch `live_translate.py` — it automatically picks up new labels from
`labels.json`.

## Limitations of this version (Stage 1)

- Recognizes **static poses only** — held still, not motions like swipes
  or circles. A future "Stage 2" upgrade would record short video clips
  instead of single snapshots and use a sequence model (LSTM) to support
  motion-based signs.
- Accuracy depends entirely on how much/good training data you collect —
  this isn't a pretrained ASL model, it learns only the gestures you teach it.
- Lighting and camera angle affect MediaPipe's hand detection reliability.
