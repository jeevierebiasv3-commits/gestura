# Setup Notes — Sign Sentence Translator

Quick-reference notes for setting up and running this project. Follow the
steps in order.

---

## What you need (system requirements)

| # | Requirement | Version | Purpose |
|---|---|---|---|
| 1 | Python | 3.9 – 3.11 | Runs everything |
| 2 | tensorflow | 2.15.0 | Builds/trains the model, runs predictions |
| 3 | mediapipe | 0.10.14 | Tracks both hands from webcam (21 points/hand) |
| 4 | protobuf | 4.25.3 | Shared dependency — pinned to avoid TF/mediapipe version clashes |
| 5 | numpy | <2 (1.26.x) | Number arrays for landmark coordinates |
| 6 | opencv-python | latest | Webcam capture, drawing HUD/subtitles, window display |
| 7 | pandas | latest | Reads/writes data.csv, label counting |
| 8 | scikit-learn | latest | Splits data, encodes labels for training |
| 9 | Webcam | — | Built-in or USB, not in use by another app |

---

## Step 1 — Check Python version

```bash
python3 --version
```

Make sure it's between 3.9 and 3.11. If not, install a compatible version
before continuing.

---

## Step 2 — Create a virtual environment

Keeps this project's packages isolated from everything else on your machine.

```bash
cd "path\to\your\project\folder"
python -m venv venv
```

**Activate it** (do this every time you open a new terminal for this project):

```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## Step 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Step 4 — Install all dependencies

With the venv active, run:

```bash
pip install -r requirements.txt
```

This installs everything in the table above using the tested, compatible
version pins.

**If you don't have `requirements.txt` handy**, install manually instead:

```bash
pip install "tensorflow==2.15.0" "mediapipe==0.10.14" "protobuf==4.25.3" "numpy<2" opencv-python pandas scikit-learn
```

---

## Step 5 — Verify the install

```bash
python -c "import cv2, mediapipe, tensorflow, pandas, numpy, sklearn; print('All good')"
```

You should see `All good` printed with no errors. If you get an error here,
something is missing or mismatched — fix it before moving on.

---

## Step 6 — Collect gesture data

```bash
python collect_data.py
```

- Type a label (e.g. `HELLO`)
- Press **SPACE** ~40-60 times to capture samples, varying hand position
  slightly each time
- Press **n** for a new gesture/label
- Press **p** to pause/resume
- Press **c** to clear the current label's samples and redo it
- Press **r** to reset any label by typing its name
- Press **q** to quit and save

Repeat until you have at least 2 (ideally 5+) gestures recorded.

### Recommended: collect a `NONE` (background) class

The model is a **closed-world softmax** — its output always sums to 100% across
the signs you trained, so it has no built-in "this isn't a sign" answer. Show it
an untrained gesture and it will still pick (and display) the *nearest* trained
sign, sometimes with high confidence. That's why an unknown hand shape still gets
"translated."

The most effective fix is to give it an explicit "not a real sign" bucket:

1. In `collect_data.py`, add a label named `NONE` (or `unknown`).
2. Burst-capture ~60-100 varied samples of **non-signs**: hands at rest, random
   or half-formed hand shapes, one hand doing nothing, hands moving between
   signs, empty-ish poses at different positions/distances.
3. Retrain. Now ambiguous input has somewhere to go, and `NONE` predictions are
   simply not shown as a sentence.

In addition, `app.py` / `live_translate.py` already reject low-confidence and
*ambiguous* predictions via two gates (tune these at the top of each file):

- `CONFIDENCE_THRESHOLD = 0.85` — the top class must be at least this confident.
- `MARGIN_THRESHOLD = 0.30` — it must also beat the 2nd-best class by this much,
  so a "torn between two signs" frame reads as nothing rather than a guess.

Raise these for fewer false positives (stricter), lower them if real signs are
being missed (more sensitive).

---


## Step 7 — Train the model

```bash
python train_model.py
```

This reads `data.csv`, trains the model, prints test accuracy, and saves
`model.h5` + `labels.json`.

**Aim for 85%+ test accuracy.** If lower, go back to Step 6 and add more
samples (especially for any gestures that look similar to each other).

---

## Step 8 — Run the live translator

```bash
python live_translate.py
```

Opens your webcam with hand tracking and shows the predicted sentence as a
subtitle at the bottom of the screen. Press **q** to quit.

---

## Step 9 — (Optional) Enable LLM sentence polishing

The motion web app (`app_sequence.py`) has a **✨ Polish** button that turns the
raw recognized signs ("Hi · Good · Morning") into one natural sentence
("Good morning!"). This runs **fully offline** through a local
[Ollama](https://ollama.com) model — no API key, no cloud, no cost.

It's optional: if Ollama isn't running the app works exactly as before and the
Polish button just shows a friendly "LLM unavailable" message.

**One-time setup:**

1. Install Ollama → https://ollama.com/download
2. Pull a small model:
   ```bash
   ollama pull llama3.2
   ```
3. Start the server (or just leave the Ollama desktop app running):
   ```bash
   ollama serve
   ```

**Use it:**

```bash
python app_sequence.py
```

Open http://localhost:5000, sign a few words, then click **✨ Polish**. The
polished sentence appears below the video (and is spoken if Voice is on).

**Optional overrides** (env vars, no code change needed):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Which pulled model to use (e.g. `phi3`, `qwen2.5`) |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening |

Quick check without the browser:

```bash
python llm.py Hi Good Morning
```

> **Note:** No new pip package is required — the Ollama call uses Python's
> built-in `urllib`. Autocomplete "next sign" chips (`suggest.py`) are a
> lightweight scaffold that becomes more useful as the sign vocabulary grows.

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | You're on too new a mediapipe version — reinstall with `pip install mediapipe==0.10.14` |
| `protobuf` version conflict errors | Reinstall the exact pinned set: Step 4 manual command above |
| Webcam won't open | Close other apps using the camera (Zoom, Teams, etc.) |
| Low training accuracy | Add more samples per gesture (40-60+), with more variation in angle/distance |
| Gestures get confused with each other | Make hand shapes more visually distinct, or add more samples for the confused pair |
| Untrained gestures still get "translated" | Expected for a softmax classifier — add a `NONE`/background class (see Step 6) and/or raise `CONFIDENCE_THRESHOLD` / `MARGIN_THRESHOLD` in `app.py` & `live_translate.py` |
| **✨ Polish** says "LLM unavailable" | Start Ollama (`ollama serve` or the desktop app). It's optional — the rest of the app runs fine without it |
| **✨ Polish** says "Model not found" | Pull the model: `ollama pull llama3.2` (or set `OLLAMA_MODEL` to one you have) |

---

## Notes for adding more gestures later

- Just re-run `collect_data.py` — it adds to the existing `data.csv`, doesn't erase previous data
- Re-run `train_model.py` afterward to retrain on the combined dataset
- No need to touch `live_translate.py` — it auto-loads whatever's in `labels.json`
