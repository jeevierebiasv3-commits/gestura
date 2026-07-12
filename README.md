# Gestura — Webcam Sign-Language Translator

Gestura turns hand gestures from your laptop's webcam into a running,
spoken sentence. [MediaPipe](https://developers.google.com/mediapipe) tracks
both hands, a neural network classifies each sign, and recognized signs are
assembled into text that is shown on screen and read aloud — no
letter-by-letter spelling involved.

There are two independent pipelines:

- **Stage 1 — Static poses** (`model.h5` / `labels.json`): a feedforward network
  classifies one held hand shape per frame. Good for signs that are a fixed pose.
- **Stage 2 — Motion signs** (`seq_model.h5` / `seq_labels.json`): a GRU sequence
  model classifies short, variable-length clips, so movement-based signs (a wave
  for "Hello", hand-from-chin for "Thank you") are recognized properly.
  **Stage 2 is the current focus** — the web app, LLM sentence polishing, and
  next-sign autocomplete are all built on it.

**Why it's accurate:** landmarks are **normalized** (each hand re-centered on the
wrist and scaled by hand size) and include depth (z) before the model sees them,
so recognition is robust to *where* your hand is in the frame and *how far* it is
from the camera. All scripts share this logic via `hand_features.py`, so there's
no train/serve skew.

## Setup

1. Install **Python 3.9–3.11** (TensorFlow doesn't support the very latest Python
   versions yet).
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   # Windows (PowerShell):
   venv\Scripts\Activate.ps1
   # macOS / Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

   This installs OpenCV (webcam), MediaPipe (hand tracking), TensorFlow (the
   models), Flask (web app), and helpers.

> The committed models (`model.h5`, `seq_model.h5`) and datasets (`data.csv`,
> `sequences/`) let you run the live translator immediately without collecting or
> training anything first.

## Usage

Each stage has three scripts that run in order: **collect → train → run live.**

### Stage 2 — Motion signs (the active pipeline)

```bash
python collect_sequences.py      # record clips  -> sequences/<LABEL>/*.npy
python train_sequence_model.py   # -> seq_model.h5 + seq_labels.json
python live_sequence.py          # desktop OpenCV window
python app_sequence.py           # web app -> open http://localhost:5000
```

### Stage 1 — Static poses

```bash
python collect_data.py    # append rows -> data.csv
python train_model.py     # -> model.h5 + labels.json
python live_translate.py  # desktop window   (app.py = web version)
```

> **Retrain whenever the label set changes.** The live scripts don't assert on
> shape, so a model whose output width no longer matches its `*_labels.json` will
> mispredict silently. After collecting new data, always retrain before running
> live.

## The web app

`app_sequence.py` is a Flask + PWA front end for Stage 2. A background thread owns
the webcam and runs the *same* segmentation and classification as
`live_sequence.py`, so desktop and web never drift apart. Features:

- Live annotated video stream and a growing sentence, updated over Server-Sent
  Events.
- **Text-to-speech** in the browser (Web Speech API — offline, client-side).
- A **"why" panel** that explains when and why a gesture was rejected.
- **Practice mode**, **next-sign autocomplete**, and optional **LLM polishing**.
- Installable as a Progressive Web App (service worker + manifest).

## Optional subsystems (degrade gracefully)

These enhance the translator but never crash it when absent:

- **LLM polish** (`llm.py`) — cleans up the raw sign sequence into a fluent
  sentence via a local [Ollama](https://ollama.com) server (`llama3.2`,
  override with `OLLAMA_HOST` / `OLLAMA_MODEL`). If Ollama isn't running it's
  simply skipped. Quick check without a webcam: `python llm.py Hi Good Morning`.
- **Autocomplete** (`suggest.py`) — a learned bigram model that suggests the
  likely next sign and re-ranks predictions with a small context prior.
- **TTS** — browser-side on the web app; `pyttsx3` on desktop (no-op if not
  installed).

## How it works under the hood

- MediaPipe detects up to 2 hands and returns 21 (x, y, z) landmarks per hand.
- **Feature vector = 126 floats**: `[Left hand 63][Right hand 63]`, each hand =
  21 landmarks × (x, y, z). A missing hand is that hand's 63 slots zeroed.
- Stage 1 feeds this 126-vector into a feedforward net (pose → label).
- Stage 2 buffers frames into a clip, resamples to a fixed length, and feeds it to
  a GRU that outputs a sign. Segmentation is **motion-gated** (it watches hand
  motion to find where a sign starts and ends) rather than a fixed sliding window,
  which is what lets variable-length and back-to-back signs work.
- A prediction is only committed if confidence and margin clear their thresholds;
  an open-set "NONE" class lets unknown gestures be ignored instead of forced into
  the nearest real sign.

## Project layout

| File | Role |
|------|------|
| `hand_features.py` | **Single source of truth** for the feature vector, normalization, and clip helpers. Shared by collection, training, and inference. |
| `live_sequence.py` | Source of truth for the Stage-2 runtime (segmentation + classification). `app_sequence.py` imports it. |
| `app_sequence.py` / `app.py` | Web apps for Stage 2 / Stage 1. |
| `train_sequence_model.py` / `train_model.py` | Trainers for each stage. |
| `collect_sequences.py` / `collect_data.py` | Data collection for each stage. |
| `llm.py`, `suggest.py` | Optional polishing and autocomplete. |

## Limitations

- Accuracy depends on how much and how varied your training data is — this is not
  a pretrained ASL model; it learns only the signs you teach it.
- Lighting and camera angle affect MediaPipe's detection reliability.
- The webcam frame is mirrored before processing; the datasets and augmentation
  account for this.

## License

[MIT](LICENSE)
