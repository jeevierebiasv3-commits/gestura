# Deploying Gestura as a Desktop App

Gestura is a **single-machine app**: the server that runs the model also owns the
webcam (`cv2.VideoCapture(0)` in `app_sequence.py`). So it isn't hosted on a
website where each visitor uses their own camera — instead it's packaged into a
**Windows app the user runs locally**, which opens in their browser at
`http://127.0.0.1:5000`. Their camera → their screen, no Python install needed.

This is the natural distribution model for this app. (A true multi-user *website*
would require porting the whole MediaPipe + GRU pipeline into the browser — a much
larger rewrite, deliberately not done here.)

## Build it

From the project root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File build_gestura.ps1
```

That installs PyInstaller into the committed `venv`, then runs `gestura.spec`.
The build takes several minutes (TensorFlow is large) and produces:

```
dist\Gestura\
  Gestura.exe          <- double-click to run
  _internal\           <- bundled Python, models, templates, static assets
```

To rebuild manually instead of via the script:

```powershell
venv\Scripts\python.exe -m pip install pyinstaller
venv\Scripts\pyinstaller.exe gestura.spec --noconfirm
```

## Run / test it

Double-click `dist\Gestura\Gestura.exe` (or run it from a terminal to watch the
log). A console window shows load progress; the browser opens automatically at
`http://127.0.0.1:5000`. **Close the console window to quit.**

First launch is slow (loading TensorFlow + the model). Grant camera access if
Windows prompts.

Learned autocomplete counts are written to `suggest_counts.json` **next to
`Gestura.exe`** (via `GESTURA_DATA_DIR`, set by `desktop_app.py`), so online
learning persists across runs. The bundled resources stay read-only.

## Ship it

Zip the **entire** `dist\Gestura\` folder and send that. The recipient unzips and
runs `Gestura.exe` — no Python, no pip, nothing to install.

- **Ollama sentence-polishing** stays optional and degrades gracefully: if the
  user doesn't have Ollama running, the "Polish" button just reports it's
  unavailable — the translator still works.
- Distribute the folder as-is, or wrap it in an installer (Inno Setup /
  NSIS) later if you want a Start-menu shortcut.

## How the packaging works (for maintainers)

- **`desktop_app.py`** — the frozen entry point. Wraps `app_sequence.py` (no
  forked logic). It `chdir`s into the PyInstaller bundle so the app's relative
  paths (`seq_model.h5`, `seq_labels.json`) resolve, points `GESTURA_DATA_DIR`
  at a writable folder next to the `.exe`, starts the `capture_loop` thread, and
  auto-opens the browser.
- **`gestura.spec`** — `collect_all` for `mediapipe` (its `.tflite`/`.binarypb`
  assets aren't auto-detected), `tensorflow`, `cv2`, and `protobuf`; bundles the
  model/label/template/static files; excludes training-only libs (`pandas`,
  `sklearn`, `matplotlib`) to shrink the output. onedir, `console=True`, no UPX
  (UPX can corrupt TensorFlow DLLs).
- **`app_sequence.py`** / **`suggest.py`** — two small frozen-aware tweaks:
  Flask's `template_folder`/`static_folder` resolve via `sys._MEIPASS`, and the
  counts file honors `GESTURA_DATA_DIR`. Both are no-ops in dev, so
  `python app_sequence.py` still works exactly as before.

## Troubleshooting

- **`Gestura.exe` opens then closes immediately** — run it from a terminal to
  read the traceback before the window vanishes.
- **`Could not open webcam`** — another app is using the camera, or Windows
  camera privacy settings block it (Settings → Privacy → Camera).
- **Blank video / template not found** — usually a missing `--add-data`; confirm
  `_internal\templates` and `_internal\static` exist in the build output.
- **Antivirus flags the `.exe`** — common false positive for PyInstaller bundles;
  sign the binary or whitelist it for wider distribution.
