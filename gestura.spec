# gestura.spec — PyInstaller build for the Gestura desktop app.
#
# Build:  venv\Scripts\pyinstaller.exe gestura.spec --noconfirm
# Output: dist\Gestura\Gestura.exe  (+ dist\Gestura\_internal\ resources)
#
# onedir (NOT onefile): TensorFlow + MediaPipe are >1 GB; onefile would
# re-extract to a temp dir on every launch (slow) and wouldn't persist the
# learned autocomplete counts. onedir gives a fast-launching folder to zip.
#
# The three landmines this spec defuses:
#   1. MediaPipe ships binary graph/model assets (.tflite/.binarypb) under
#      mediapipe/modules that PyInstaller does NOT auto-detect -> collect_all.
#   2. TensorFlow's dynamic imports / data likewise need collect_all.
#   3. Our own runtime resources (the GRU model, labels, templates, static PWA
#      files) must be bundled and found at their relative paths.

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# --- heavy third-party packages with hidden/dynamic pieces ---
for pkg in ("mediapipe", "tensorflow", "cv2", "google.protobuf"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# --- our runtime data: Stage-2 model + labels, and the web assets ---
# (also bundle the Stage-1 files so nothing 404s if referenced.)
datas += [
    ("seq_model.h5", "."),
    ("seq_labels.json", "."),
    ("model.h5", "."),
    ("labels.json", "."),
    ("templates", "templates"),
    ("static", "static"),
]

# our own modules imported dynamically / by string are picked up normally, but
# list the ones PyInstaller's static analysis can miss to be safe.
hiddenimports += [
    "hand_features", "live_sequence", "llm", "suggest", "app_sequence",
    "engineio.async_drivers.threading",
]

block_cipher = None

a = Analysis(
    ["desktop_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # keep training-only / desktop-only libs out to shrink the build. NOTE:
    # matplotlib is NOT excluded — MediaPipe's drawing_utils imports
    # matplotlib.pyplot at load time, so excluding it crashes the app on
    # startup (ModuleNotFoundError). tkinter stays in for matplotlib's backend.
    excludes=["pandas", "sklearn", "scikit-learn", "pyttsx3",
              "PyQt5", "PySide2", "notebook", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Gestura",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX + TF DLLs can corrupt; keep off.
    console=True,            # keep a console so users see load progress / errors.
    disable_windowed_traceback=False,
    icon="static/gestura.ico" if __import__("os").path.exists("static/gestura.ico") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Gestura",
)
