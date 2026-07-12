import cv2
import numpy as np
import json
import threading
import time
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify, stream_with_context
from tensorflow.keras.models import load_model

import mediapipe as mp
import hand_features as hf

MODEL_FILE = "model.h5"
LABELS_FILE = "labels.json"
CONFIDENCE_THRESHOLD = 0.85   # top class must be at least this confident
MARGIN_THRESHOLD = 0.30       # ...AND beat the 2nd-best class by this much
SMOOTHING_WINDOW = 8
LANDMARK_SMOOTHING = 0.5
DROPOUT_TOLERANCE = 4

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = hf.make_hands()

model = load_model(MODEL_FILE)
with open(LABELS_FILE) as f:
    label_map = json.load(f)

app = Flask(__name__)

frame_lock = threading.Lock()
latest_frame = None
latest_prediction = {"text": "", "confidence": 0.0, "hands": 0}
gesture_labels = list(label_map.values())


class LandmarkSmoother:
    """Exponential moving average over the raw 126-number landmark vector, plus
    dropout tolerance: if a hand briefly isn't detected we hold its last known
    position for a few frames instead of snapping to zeros, reducing flicker."""

    def __init__(self, alpha=LANDMARK_SMOOTHING, dropout_tolerance=DROPOUT_TOLERANCE):
        self.alpha = alpha
        self.dropout_tolerance = dropout_tolerance
        self.smoothed = None
        self.missing_left = 0
        self.missing_right = 0

    def update(self, raw_landmarks):
        ph = hf.PER_HAND
        left_raw = raw_landmarks[:ph]
        right_raw = raw_landmarks[ph:]
        left_present = any(v != 0.0 for v in left_raw)
        right_present = any(v != 0.0 for v in right_raw)

        if self.smoothed is None:
            self.smoothed = list(raw_landmarks)
            self.missing_left = 0 if left_present else self.dropout_tolerance + 1
            self.missing_right = 0 if right_present else self.dropout_tolerance + 1
            return list(self.smoothed), left_present, right_present

        prev_left = self.smoothed[:ph]
        prev_right = self.smoothed[ph:]

        if left_present:
            self.missing_left = 0
            new_left = [self.alpha * p + (1 - self.alpha) * r
                        for p, r in zip(prev_left, left_raw)]
        else:
            self.missing_left += 1
            new_left = prev_left if self.missing_left <= self.dropout_tolerance else [0.0] * ph

        if right_present:
            self.missing_right = 0
            new_right = [self.alpha * p + (1 - self.alpha) * r
                         for p, r in zip(prev_right, right_raw)]
        else:
            self.missing_right += 1
            new_right = prev_right if self.missing_right <= self.dropout_tolerance else [0.0] * ph

        self.smoothed = new_left + new_right
        effective_left = left_present or self.missing_left <= self.dropout_tolerance
        effective_right = right_present or self.missing_right <= self.dropout_tolerance
        return list(self.smoothed), effective_left, effective_right


def capture_loop():
    global latest_frame, latest_prediction
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    smoother = LandmarkSmoother()
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        raw_landmarks = hf.extract_raw_hands(results)
        smoothed_landmarks, left_ok, right_ok = smoother.update(raw_landmarks)
        hands_visible = left_ok or right_ok

        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        pred_conf = 0.0
        if hands_visible:
            features = hf.normalize(smoothed_landmarks).reshape(1, -1)
            probs = model.predict(features, verbose=0)[0]
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            # Reject untrained/ambiguous gestures: require both high confidence
            # AND a clear margin over the runner-up. A softmax always picks some
            # known class, so without this an unseen sign gets mislabeled.
            if probs.size >= 2:
                second_conf = float(np.partition(probs, -2)[-2])
            else:
                second_conf = 0.0
            margin = best_conf - second_conf
            if best_conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                recent_predictions.append(label_map[str(best_idx)])
                pred_conf = best_conf
            else:
                recent_predictions.append(None)
        else:
            recent_predictions.append(None)

        pred_text = ""
        valid = [p for p in recent_predictions if p is not None]
        if len(valid) >= SMOOTHING_WINDOW // 2:
            pred_text = Counter(valid).most_common(1)[0][0].replace("_", " ")

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()
            latest_prediction = {
                "text": pred_text,
                "confidence": round(pred_conf * 100),
                "hands": hand_count,
            }
    cap.release()


def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            frame_data = latest_frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html", gestures=gesture_labels)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/prediction")
def prediction():
    def generate():
        while True:
            with frame_lock:
                pred = dict(latest_prediction)
            yield f"data: {json.dumps(pred)}\n\n"
            time.sleep(0.1)
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/gestures")
def gestures():
    return jsonify(gesture_labels)


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
