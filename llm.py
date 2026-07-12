"""
llm.py  —  offline sentence polishing via a local Ollama model
--------------------------------------------------------------
The sign recognizer produces a sequence of raw glosses ("Hi", "Good", "Morning").
This turns that word-salad into one natural English sentence ("Good morning!")
by asking a local LLM served by Ollama (https://ollama.com).

Everything stays on-device: no API key, no cloud, no per-call cost. If Ollama
isn't running (or the model isn't pulled) polish() fails *gracefully* — it never
raises into the request handler, it returns {"ok": False, "error": ...} so the
web UI can show a friendly message and the rest of the app keeps working.

Setup (one time):
    1. install Ollama         -> https://ollama.com/download
    2. pull a small model     -> ollama pull llama3.2
    3. start the server       -> ollama serve   (or the desktop app)

Override the defaults with env vars:
    OLLAMA_HOST   (default http://localhost:11434)
    OLLAMA_MODEL  (default llama3.2)
"""

import os
import json
import urllib.request
import urllib.error

# --- configuration (env-overridable so no code change is needed to retarget) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
REQUEST_TIMEOUT = 15          # seconds — keep the UI from ever hanging on a stall

# Instruction sent to the model. Kept short and firm so small local models return
# just the sentence (not an explanation or a list of options).
_SYSTEM = (
    "You help a deaf user communicate. You are given sign-language glosses that "
    "were recognized in order. Rewrite them as ONE short, natural, grammatical "
    "English sentence that conveys what the signer most likely means. Fix word "
    "order, add small connecting words, and punctuate. Do not add information "
    "that isn't implied. Reply with ONLY the sentence — no quotes, no preamble, "
    "no explanation."
)


def _build_prompt(words):
    glosses = ", ".join(words)
    return (
        f"{_SYSTEM}\n\n"
        f"Glosses (in order): {glosses}\n"
        f"Sentence:"
    )


def _clean(text):
    """Strip the surrounding quotes / stray whitespace small models often add."""
    text = (text or "").strip()
    # models sometimes echo a "Sentence:" prefix
    for prefix in ("Sentence:", "sentence:", "Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # collapse to the first line — we only ever want a single sentence. Do this
    # BEFORE unquoting so a trailing extra line can't defeat the quote-strip.
    if text:
        text = text.splitlines()[0].strip()
    # drop matching surrounding quotes the model may have added
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def polish(words):
    """Turn a list of recognized sign glosses into one natural sentence.

    Returns a dict the Flask route can jsonify directly:
        {"ok": True,  "polished": "Good morning!"}
        {"ok": False, "error": "Nothing to polish yet."}
    Never raises — a down/absent Ollama becomes a friendly error string.
    """
    words = [w.strip() for w in (words or []) if w and w.strip()]
    if not words:
        return {"ok": False, "error": "Nothing to polish yet."}

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": _build_prompt(words),
        "stream": False,
        # low temperature — we want a faithful rewrite, not creative writing
        "options": {"temperature": 0.2},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # server reachable but unhappy — most often the model isn't pulled
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", "")
        except Exception:
            pass
        if "not found" in detail.lower() or e.code == 404:
            return {"ok": False,
                    "error": f"Model '{OLLAMA_MODEL}' not found — run: "
                             f"ollama pull {OLLAMA_MODEL}"}
        return {"ok": False, "error": f"LLM error ({e.code}). {detail}".strip()}
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        # Ollama not running / unreachable — the common offline case
        return {"ok": False,
                "error": "LLM unavailable — start Ollama (ollama serve) to polish."}
    except Exception as e:  # last-resort guard; never crash the request
        return {"ok": False, "error": f"Polish failed: {e}"}

    polished = _clean(body.get("response", ""))
    if not polished:
        return {"ok": False, "error": "The model returned an empty response."}
    return {"ok": True, "polished": polished}


if __name__ == "__main__":
    # quick manual check:  python llm.py Hi Good Morning
    import sys
    demo = sys.argv[1:] or ["Hi", "Good", "Morning"]
    print(f"input : {demo}")
    print(f"output: {polish(demo)}")
