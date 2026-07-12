# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A webcam sign-language translator: MediaPipe tracks both hands, a neural network
classifies the gesture, and recognized signs are assembled into a running
sentence (shown on screen and spoken aloud). Two independent pipelines coexist:

- **Stage 1 — static poses** (`model.h5` / `labels.json`): a feedforward net
  classifies one held hand shape per frame.
- **Stage 2 — motion signs** (`seq_model.h5` / `seq_labels.json`): a GRU
  sequence model classifies short variable-length clips, so movement-based signs
  (wave = "Hello", hand-from-chin = "Thank you") work. **Stage 2 is the current
  focus** — new features (web app, LLM polish, autocomplete) are built on it.

Each stage has three scripts that run in order: **collect → train → run live.**

## Environment & commands

There is no build system, test suite, or linter. Everything runs through the
committed `venv/`. Use it directly (do not assume a global `python`):

```bash
./venv/Scripts/python.exe <script>.py     # Git Bash
venv\Scripts\python.exe <script>.py        # PowerShell / cmd
```

Stage 2 (motion — the active pipeline):
```
python collect_sequences.py      # record clips -> sequences/<LABEL>/*.npy
python train_sequence_model.py   # -> seq_model.h5 + seq_labels.json
python live_sequence.py          # desktop OpenCV window
python app_sequence.py           # web app, then open http://localhost:5000
```

Stage 1 (static):
```
python collect_data.py    # append rows -> data.csv
python train_model.py     # -> model.h5 + labels.json
python live_translate.py  # desktop window   (app.py = web version)
```

Quick manual check of the LLM path without a webcam:
```
python llm.py Hi Good Morning
```

**A model must be retrained whenever its label set changes** — the live scripts
assert nothing, so a model whose output width no longer matches its `*_labels.json`
will mispredict silently. After collecting new data, always retrain before running
live.

## Architecture — the parts that span files

**`hand_features.py` is the single source of truth** and the most important file
to understand. Collection, training, and inference all import it so the *exact
same* feature logic runs everywhere (no train/serve skew). If you change a
feature-layout or normalization detail here, every model becomes stale and must
be retrained. Key facts encoded here:

- **Feature vector = 126 floats**: `[Left hand 63][Right hand 63]`, each hand =
  21 landmarks × (x, y, z). A missing hand is that hand's 63 slots zeroed.
  Handedness (`extract_raw_hands`) decides which slot a detected hand fills.
- **`normalize()`** re-centers each hand on the wrist and scales by wrist→middle-MCP
  distance → position- and distance-invariance. Rotation is *deliberately kept*
  (orientation is meaningful in signing). This normalization is why recognition is
  robust to where/how-far the hand is; it was the fix for the old raw-xy inaccuracy.
- **Raw (un-normalized) landmarks are what's stored on disk** (`data.csv`, the
  `.npy` clips). Normalization + augmentation happen at train/infer time, never at
  collection time.
- Stage-2 clip helpers: `resample_sequence` (any-length clip → fixed `SEQ_LEN=45`
  frames via interpolation), `normalize_sequence`, `frame_motion` (used for live
  segmentation), and `augment`/`augment_sequence`.

**`live_sequence.py` is the source of truth for the Stage-2 runtime pipeline**,
not just a demo — `app_sequence.py` imports its constants and functions
(`classify_segment_debug`, the motion thresholds, `IGNORE_LABELS`, the shared
MediaPipe `hands` instance) so desktop and web can never drift apart. When
changing motion behavior, edit it here. Two mechanisms live here:

- **Motion-gated segmentation** (not a fixed sliding window): watch
  `frame_motion`; when it exceeds `MOTION_START` begin buffering a segment; when
  motion stays below `MOTION_STOP` for `STOP_FRAMES`, the sign has ended → resample
  → normalize → classify. This is what lets variable-length and back-to-back
  (pause-separated) signs work.
- **Acceptance gates**: a prediction is only committed if `conf >=
  CONFIDENCE_THRESHOLD` **and** `margin (top − 2nd) >= MARGIN_THRESHOLD`. These are
  intentionally relaxed while the dataset is small; the docstring says to raise
  them once classes have ~30–50 balanced clips. `classify_segment_debug` returns
  *why* a segment was rejected so the web "why panel" can surface it.

**`IGNORE_LABELS`** is the open-set reject mechanism: labels like
`"this isn't a hand sign"` are recognized but never displayed/appended. Training
(`train_sequence_model.py`) actively warns if no such NONE/background class
exists, because without one, unknown gestures get forced into the nearest real
sign. When adding signs, keep a NONE class in the dataset.

### Web app data flow (`app_sequence.py`)

Flask serves Stage 2. A background `capture_loop` thread owns the webcam and runs
the *same* segmentation/classification as `live_sequence.py`, writing into a
shared `state` dict under `frame_lock`. Routes read that state:

- `/video_feed` — MJPEG stream of annotated frames.
- `/state` — Server-Sent Events pushing the running sentence + live status;
  `speak_seq` increments when a new word should be spoken (browser Web Speech API
  does the TTS, offline, client-side).
- `/clear` (POST) sets a one-way flag consumed by the loop; `/polish` (POST) reads
  the **server-side** sentence (not client input) and calls `llm.polish`.
- It's a PWA — `/service-worker.js` and `/manifest.webmanifest` are served from
  root scope with special headers; `make_icons.py` regenerates the icons.

### Optional, degrade-gracefully subsystems

These never crash the core translator when absent — mirror that pattern in new code:

- **`llm.py`** — offline sentence polishing via a local **Ollama** server
  (`llama3.2`, env-overridable `OLLAMA_HOST`/`OLLAMA_MODEL`). If Ollama isn't
  running, `polish()` returns `{"ok": False, "error": ...}`, never raises.
- **`suggest.py`** — next-sign autocomplete. Currently a hand-seeded transition
  table + popularity fallback; structured so a learned bigram table drops into
  `TRANSITIONS` without touching callers.
- **`apply_context_prior` in `live_sequence.py`** — a wired-but-identity hook for
  re-ranking softmax by the previous sign; left a no-op while the vocab is tiny.
- **TTS** (desktop, `pyttsx3` via the `Speaker` class) — no-op if not installed;
  runs on a worker thread so speech never stalls the video loop.

## Conventions

- Keep collection/training/inference feature logic in `hand_features.py` — never
  duplicate normalization or the 126-vector layout in a script.
- Desktop and web Stage-2 code must share `live_sequence.py`'s logic; don't fork
  the motion pipeline into `app_sequence.py`.
- The webcam frame is mirrored (`cv2.flip`) before processing; MediaPipe
  handedness is relative to that mirrored view (the `mirror()` augmentation exists
  to tolerate the resulting flips).
