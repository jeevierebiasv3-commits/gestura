"""
desktop_app.py  —  packaged-desktop entry point for Gestura (Stage 2 web app)
-----------------------------------------------------------------------------
This is the launcher PyInstaller freezes into Gestura.exe. It is a thin wrapper
around app_sequence.py so the desktop build and the dev server run the EXACT
same Flask app / recognition pipeline — no forked logic (per the project rule
that desktop and web must not drift).

What it adds on top of `python app_sequence.py`:

  1. Frozen path fix — a PyInstaller onedir build unpacks its bundled data
     (models, templates, static) under sys._MEIPASS. We chdir there BEFORE
     importing app_sequence, because that import loads seq_model.h5 /
     seq_labels.json by *relative* path at import time.

  2. A writable data dir — the bundle is read-only, so we point
     GESTURA_DATA_DIR at a folder next to the .exe. That's where suggest.py
     persists its learned autocomplete counts, so online learning survives
     restarts instead of silently failing on a read-only path.

  3. Convenience — opens the user's browser at the app automatically, so a
     double-click on the .exe "just works".

In dev this file also runs fine:  python desktop_app.py
"""

import os
import sys
import threading
import webbrowser

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def _bundle_dir():
    """Folder holding the bundled resources (models/templates/static)."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _writable_data_dir():
    """A folder we can WRITE to that survives restarts. Next to the .exe when
    frozen; the source folder in dev."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. resolve paths BEFORE importing the app (import loads the model by
    #    relative path, and suggest.py reads GESTURA_DATA_DIR at import).
    os.chdir(_bundle_dir())
    data_dir = _writable_data_dir()
    os.environ.setdefault("GESTURA_DATA_DIR", data_dir)

    # 2. now it's safe to import the Flask app + capture loop.
    print("[Gestura] loading model… (first launch takes a few seconds)")
    import app_sequence as appmod

    # 3. start the webcam / recognition thread (same as app_sequence __main__).
    t = threading.Thread(target=appmod.capture_loop, daemon=True)
    t.start()

    # 4. open the browser once the server has had a moment to bind the port.
    threading.Timer(2.0, lambda: webbrowser.open(URL)).start()

    print(f"[Gestura] open {URL} in your browser (opening automatically)…")
    print("[Gestura] close this window to quit.")
    # threaded=True so the MJPEG stream + SSE + normal routes serve concurrently.
    appmod.app.run(host=HOST, port=PORT, debug=False, threaded=True,
                   use_reloader=False)


if __name__ == "__main__":
    main()
