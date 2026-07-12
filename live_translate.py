"""
live_translate.py
-------------------
Runs your webcam live, tracks both hands, and predicts which trained
sentence-gesture is being shown, displaying the result as a movie-style
subtitle at the bottom of the screen.

WHAT'S NEW (accuracy overhaul)
- Uses the shared hand_features.py for landmark extraction (with depth z) and
  the wrist-centered + scale-normalized features. This is identical to what
  training used, so there is no train/serve skew, and predictions no longer
  depend on where your hand sits in the frame or how far it is from the camera.

HOW TO USE
1. Run collect_data.py and train_model.py first (need model.h5 + labels.json).
2. Run: python live_translate.py
3. Show a trained gesture; the predicted sentence appears as a subtitle once
   the model is confident enough. Press 'q' to quit.
"""

import cv2
import numpy as np
import json
from collections import deque, Counter
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


class LandmarkSmoother:
    """EMA over the raw 126-number landmark vector + dropout tolerance so brief
    tracking losses don't snap a hand to zeros (reduces prediction flicker)."""

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


def draw_subtitle(frame, text):
    h, w = frame.shape[:2]
    if not text:
        text = ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_width = w - 60

    font_scale = 1.3
    thickness = 3
    min_scale = 0.5
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    while text_size[0] > max_text_width and font_scale > min_scale:
        font_scale -= 0.05
        thickness = max(2, int(font_scale * 2.2))
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_w, text_h = text_size
    bar_height = text_h + 50

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_x = max(20, (w - text_w) // 2)
    text_y = h - (bar_height - text_h) // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    model = load_model(MODEL_FILE)
    with open(LABELS_FILE) as f:
        label_map = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    current_sentence = ""
    smoother = LandmarkSmoother()

    WINDOW_NAME = "Sign Sentence Translator"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    print("Press 'q' to quit.")

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

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
            else:
                recent_predictions.append(None)
        else:
            recent_predictions.append(None)

        valid = [p for p in recent_predictions if p is not None]
        if len(valid) >= SMOOTHING_WINDOW // 2:
            current_sentence = Counter(valid).most_common(1)[0][0]
        else:
            current_sentence = ""

        display_text = current_sentence.replace("_", " ") if current_sentence else ""
        draw_subtitle(frame, display_text)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
