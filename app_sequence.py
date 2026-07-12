"""
app_sequence.py  (Stage 2 on the web — motion signs)
----------------------------------------------------
The browser version of live_sequence.py. Same MOTION-GATED SEGMENTATION and
sentence-building as the desktop app, but served over Flask so anyone can open
it in a browser — no Python/OpenCV window needed on the viewer's side.

  - the webcam frames (with hand landmarks drawn) stream as MJPEG to the page
  - a Server-Sent-Events stream pushes the running sentence + live status
  - the browser speaks each newly recognized sign with the Web Speech API
    (offline, on the viewer's own device — nothing to install)

All the recognition logic (motion thresholds, segmentation, the confidence +
margin gates, IGNORE_LABELS) is imported from live_sequence.py so the desktop
and web apps can never drift apart.

Run:  python app_sequence.py   then open  http://localhost:5000
"""

import cv2
import os
import sys
import json
import time
import threading
from collections import deque

import numpy as np
from flask import (Flask, render_template, Response, jsonify, request,
                   stream_with_context, send_from_directory, make_response)
from tensorflow.keras.models import load_model

import hand_features as hf
import live_sequence as ls   # single source of truth for the motion pipeline
import llm                   # offline sentence polishing via local Ollama
import suggest               # next-sign autocomplete (scaffold)


def _res(rel):
    """Resolve a bundled resource (templates/, static/) both in dev and when
    frozen by PyInstaller. Frozen builds unpack data under sys._MEIPASS; in dev
    it's just the folder next to this file."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# reuse the exact same MediaPipe drawing helpers + Hands instance
mp_drawing = ls.mp_drawing
mp_styles = ls.mp_styles
mp_hands_conn = ls.mp.solutions.hands.HAND_CONNECTIONS

model = load_model(ls.MODEL_FILE)
with open(ls.LABELS_FILE) as f:
    label_map = json.load(f)

# real sign labels (minus the "not a hand sign" reject class) for autocomplete
VOCAB = [v for v in label_map.values()
         if v.strip().lower() not in ls.IGNORE_LABELS]

app = Flask(__name__, template_folder=_res("templates"),
            static_folder=_res("static"))

frame_lock = threading.Lock()
latest_frame = None
# everything the page needs to render, refreshed every processed frame
state = {
    "sentence": [],       # [{"word": str, "conf": int(0-100)}] most-recent last
    "collecting": False,  # a sign is being recorded right now
    "seg_frames": 0,      # how many frames long the in-progress sign is
    "hands": 0,           # hands visible this frame
    "last_word": "",      # most recent recognized sign (status line)
    "last_conf": 0,
    "speak_seq": 0,       # bumps each time a NEW word should be spoken
    "speak_word": "",     # the word tied to the current speak_seq
    "suggestions": [],    # next-sign autocomplete chips (scaffold)
    "last_committed": "", # previous accepted sign — context for the prior hook
    "last_debug": None,   # {top, conf, margin, accepted, reason, seq} — why-panel
    "motion": 0.0,        # live frame_motion — drives the client "listening" meter
    "fps": 0.0,           # smoothed processing rate — client HUD
}
# one-way control flags set by HTTP routes, consumed by the capture loop
control = {"clear": False, "practice": False}


def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    preroll = deque(maxlen=ls.PREROLL)
    segment = []
    collecting = False
    still_count = 0
    start_count = 0                # consecutive frames of motion >= MOTION_START
    prev_raw = None

    sentence = []            # list of (word, conf_float)
    last_added_at = 0.0
    last_activity = time.time()
    speak_seq = 0
    speak_word = ""
    suggestions = []         # next-sign autocomplete for the current sentence
    last_committed = ""      # previous accepted sign (context for the prior hook)
    last_debug = None        # diagnostics of the most recent classified segment
    last_debug_seq = 0       # bumps each classify so the UI shows fresh results
    fps = 0.0                # exponential moving average of the processing rate
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # smoothed FPS for the client HUD (EMA so it doesn't jitter)
        now_t = time.time()
        dt = now_t - prev_t
        prev_t = now_t
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else fps * 0.9 + inst * 0.1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = ls.hands.process(rgb)

        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands_conn,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        raw = hf.extract_raw_hands(results)
        hand_present = any(v != 0.0 for v in raw)
        motion = hf.frame_motion(prev_raw, raw) if prev_raw is not None else 0.0

        # --- motion-gated segmentation (identical to live_sequence.py) ---
        finished = None
        if not collecting:
            preroll.append(raw)
            # need START_FRAMES consecutive above-threshold frames, not one spike
            if hand_present and motion >= ls.MOTION_START:
                start_count += 1
            else:
                start_count = 0
            if start_count >= ls.START_FRAMES:
                collecting = True
                segment = list(preroll)
                still_count = 0
                start_count = 0
        else:
            last_activity = time.time()
            segment.append(raw)
            if motion < ls.MOTION_STOP or not hand_present:
                still_count += 1
            else:
                still_count = 0
            if still_count >= ls.STOP_FRAMES or len(segment) >= ls.MAX_SEG:
                finished = segment
                collecting = False
                segment = []
                preroll.clear()

        if finished is not None and len(finished) >= ls.MIN_SEG:
            # During practice, grade the RAW recognizer: pass no prev_label so the
            # context prior can't nudge a score toward whatever sign was practiced
            # just before (that's not a real conversational sequence).
            prev_for_prior = None if control["practice"] else last_committed
            d = ls.classify_segment_debug(
                model, label_map, finished, prev_label=prev_for_prior)
            label = d["top"] if d["accepted"] else None
            conf = d["conf"]
            is_ignore = d["top"].strip().lower() in ls.IGNORE_LABELS
            # remember the last outcome so the UI can show WHY a sign didn't land
            last_debug = {
                "top": d["top"], "conf": round(d["conf"] * 100),
                "margin": round(d["margin"] * 100),
                "accepted": bool(d["accepted"] and not is_ignore),
                "reason": ("not a sign" if (d["accepted"] and is_ignore)
                           else d["reason"]),
                "seq": last_debug_seq + 1,
            }
            last_debug_seq += 1
            if label is not None and not is_ignore:
                sign = label.replace("_", " ")
                now = time.time()
                last_activity = now
                is_repeat = bool(sentence) and sentence[-1][0] == sign
                if not is_repeat or (now - last_added_at) >= ls.REPEAT_GAP_SECONDS:
                    # learn this (prev -> current) transition from real signing
                    # (never during practice — those aren't real sequences)
                    if last_committed and not control["practice"]:
                        suggest.record_transition(last_committed, sign)
                    sentence.append((sign, conf))
                    del sentence[:-ls.MAX_SENTENCE]
                    last_added_at = now
                    speak_seq += 1          # tell the browser to speak this word
                    speak_word = sign
                else:
                    sentence[-1] = (sign, conf)
                last_committed = sign
                # refresh next-sign suggestions from the sentence so far
                suggestions = suggest.next_signs(
                    [w for w, _ in sentence], VOCAB)

        # auto-clear after an idle pause (only when not signing)
        if sentence and not collecting and \
                (time.time() - last_activity) > ls.AUTO_CLEAR_SECONDS:
            sentence.clear()
            suggestions = []
            last_committed = ""

        # honor a clear request from the web UI
        if control["clear"]:
            sentence.clear()
            suggestions = []
            last_committed = ""
            control["clear"] = False

        last_word = sentence[-1][0] if sentence else ""
        last_conf = round(sentence[-1][1] * 100) if sentence else 0

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()
            state["sentence"] = [
                {"word": w, "conf": round(c * 100)} for w, c in sentence
            ]
            state["collecting"] = collecting
            state["seg_frames"] = len(segment) if collecting else 0
            state["hands"] = hand_count
            state["last_word"] = last_word
            state["last_conf"] = last_conf
            state["speak_seq"] = speak_seq
            state["speak_word"] = speak_word
            state["suggestions"] = suggestions if sentence else []
            state["last_committed"] = last_committed
            state["last_debug"] = last_debug
            state["motion"] = round(float(motion), 4)
            state["fps"] = round(float(fps), 1)

        prev_raw = raw

    cap.release()


def gen_frames():
    while True:
        with frame_lock:
            frame_data = latest_frame
        if frame_data is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               frame_data + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("sequence.html",
                           gestures=list(label_map.values()))


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def state_stream():
    def generate():
        while True:
            with frame_lock:
                payload = json.dumps(state)
            yield f"data: {payload}\n\n"
            time.sleep(0.1)
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream")


@app.route("/config")
def config():
    # expose the motion thresholds so the client "listening" meter can scale
    # against the SAME numbers the segmentation loop uses (no hard-coding).
    return jsonify({
        "motion_start": ls.MOTION_START,
        "motion_stop": ls.MOTION_STOP,
        "max_seg": ls.MAX_SEG,
        "min_seg": ls.MIN_SEG,
        # the "not a real sign" labels — so the client can filter the practice
        # vocabulary against the SAME reject set the server uses (see VOCAB above)
        "ignore_labels": sorted(ls.IGNORE_LABELS),
    })


@app.route("/clear", methods=["POST"])
def clear():
    control["clear"] = True
    return jsonify({"ok": True})


@app.route("/practice", methods=["POST"])
def practice():
    # the client flips this on entering/leaving practice mode. While on, the
    # capture loop grades on the raw recognizer (no context prior) and doesn't
    # learn transitions from practice attempts.
    control["practice"] = bool((request.get_json(silent=True) or {}).get("active"))
    return jsonify({"ok": True, "practice": control["practice"]})


@app.route("/polish", methods=["POST"])
def polish():
    # read the authoritative sentence from server state (don't trust the client)
    with frame_lock:
        words = [w["word"] for w in state["sentence"]]
    return jsonify(llm.polish(words))


# --- PWA: serve the service worker + manifest from the site root scope ---
@app.route("/service-worker.js")
def service_worker():
    resp = make_response(send_from_directory(app.static_folder, "service-worker.js"))
    resp.headers["Content-Type"] = "application/javascript"
    # allow a root-served SW to control the whole origin, and never stale-cache it
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/manifest.webmanifest")
def manifest():
    resp = make_response(send_from_directory(app.static_folder, "manifest.webmanifest"))
    resp.headers["Content-Type"] = "application/manifest+json"
    return resp


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
