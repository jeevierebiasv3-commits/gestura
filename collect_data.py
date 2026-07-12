"""
collect_data.py
----------------
Records hand-landmark snapshots from your webcam and saves them as labeled
training data for sentence gestures (using BOTH hands).

WHAT'S NEW (accuracy overhaul)
- Landmarks now include depth (z): 21 points x 2 hands x 3 coords = 126 numbers
  per row (was 84). Feature extraction is shared with training/inference via
  hand_features.py, so what you collect matches exactly what the model sees.
- CONTINUOUS CAPTURE: hold 'b' (or toggle with SPACE-hold) to auto-record ~5
  samples/sec, which makes it fast to gather 40-60 naturally varied samples.
- If an old-format data.csv (84 columns, no z) is found, it is archived to
  data_legacy_xyonly.csv and a fresh 126-column file is started.

HOW TO USE
1. Run: python collect_data.py
2. Type a label name when prompted (e.g. "I_AM_HUNGRY"). Use underscores.
3. A webcam window opens. Get your hands into the gesture pose.
4. Press SPACE to capture one sample, OR hold 'b' to burst-capture continuously
   while you slowly vary hand position/angle/distance (~40-60 per gesture).
5. Press 'n' to finish this label and type a new one.
6. Press 'p' to pause/resume.
7. Press 'c' to clear ALL samples for the CURRENT label.
8. Press 'r' to reset/clear samples for ANY label by typing its name.
9. Press 'q' to quit.

CONTROLS (also shown on screen):
  SPACE = capture one sample     b (hold) = burst capture
  n = new label   p = pause/resume   c = clear current   r = reset label   q = quit
"""

import cv2
import csv
import os
import time
import shutil
import pandas as pd

import mediapipe as mp
import hand_features as hf

DATA_FILE = "data.csv"
LEGACY_FILE = "data_legacy_xyonly.csv"
BURST_INTERVAL = 0.2  # seconds between auto-captures while burst key held (~5/sec)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = hf.make_hands()
CSV_HEADER = hf.csv_header()


def init_csv():
    """Create data.csv with the current header, archiving any old-format file."""
    if os.path.exists(DATA_FILE):
        try:
            existing_cols = list(pd.read_csv(DATA_FILE, nrows=0).columns)
        except pd.errors.EmptyDataError:
            existing_cols = []
        if existing_cols and existing_cols != CSV_HEADER:
            shutil.move(DATA_FILE, LEGACY_FILE)
            print(f"Old-format data.csv archived to {LEGACY_FILE} "
                  f"({len(existing_cols)} cols). Starting fresh 126-col file.")
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def get_label_counts():
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        df = pd.read_csv(DATA_FILE)
        if "label" not in df.columns or len(df) == 0:
            return {}
        return df["label"].value_counts().to_dict()
    except pd.errors.EmptyDataError:
        return {}


def clear_label(label_to_clear):
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        return 0
    if "label" not in df.columns:
        return 0
    before = len(df)
    df = df[df["label"] != label_to_clear]
    removed = before - len(df)
    df.to_csv(DATA_FILE, index=False)
    return removed


def save_sample(results, label):
    row = hf.extract_raw_hands(results) + [label]
    with open(DATA_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def hands_present(results):
    return bool(results and results.multi_hand_landmarks)


def draw_text_outlined(frame, text, pos, font_scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_hud(frame, label, count, paused, bursting, label_counts):
    h, w = frame.shape[:2]

    if paused:
        status_text, status_color = "PAUSED", (0, 165, 255)
    elif bursting:
        status_text, status_color = "BURST", (0, 255, 255)
    else:
        status_text, status_color = "RECORDING", (0, 255, 0)
    draw_text_outlined(frame, status_text, (w - 200, 30), 0.8, status_color, 2)

    draw_text_outlined(frame, f"Label: {label}", (10, 30), 0.8, (0, 255, 0), 2)
    draw_text_outlined(frame, f"Samples this label: {count}", (10, 60), 0.65, (0, 255, 0), 2)

    total = sum(label_counts.values())
    draw_text_outlined(frame, f"Total samples: {total}  |  Gestures: {len(label_counts)}",
                        (10, 85), 0.55, (200, 200, 200), 1)

    sidebar_x = w - 230
    draw_text_outlined(frame, "Collected so far:", (sidebar_x + 10, 122), 0.5, (255, 255, 255), 1)
    y = 148
    for lbl, cnt in sorted(label_counts.items()):
        color = (0, 255, 255) if lbl == label else (220, 220, 220)
        text = f"{lbl}: {cnt}"
        if len(text) > 26:
            text = text[:24] + ".."
        draw_text_outlined(frame, text, (sidebar_x + 10, y), 0.45, color, 1)
        y += 20
        if y > h - 70:
            draw_text_outlined(frame, "...", (sidebar_x + 10, y), 0.45, (200, 200, 200), 1)
            break

    cv2.rectangle(frame, (0, h - 30), (w, h), (40, 40, 40), -1)
    controls = "SPACE=capture  b(hold)=burst  n=new label  p=pause  c=clear  r=reset  q=quit"
    cv2.putText(frame, controls, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    init_csv()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check it's not in use by another app.")
        return

    label = input("Enter label for this gesture (e.g. I_AM_HUNGRY): ").strip()
    label_counts = get_label_counts()
    count = label_counts.get(label, 0)
    paused = False
    last_burst = 0.0

    WINDOW_NAME = "Collect Sign Data"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    print("\nControls: SPACE=capture | b(hold)=burst | n=new label | p=pause | "
          "c=clear current | r=reset label | q=quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural feel
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = None
        if not paused:
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

        key = cv2.waitKey(1) & 0xFF
        bursting = (key == ord("b")) and not paused

        # burst capture: auto-save on an interval while 'b' is held and a hand is visible
        if bursting and hands_present(results):
            now = time.time()
            if now - last_burst >= BURST_INTERVAL:
                save_sample(results, label)
                count += 1
                label_counts[label] = count
                last_burst = now

        draw_hud(frame, label, count, paused, bursting, label_counts)
        cv2.imshow(WINDOW_NAME, frame)

        if key == ord(" ") and not paused:
            if hands_present(results):
                save_sample(results, label)
                count += 1
                label_counts[label] = count
                print(f"Captured sample #{count} for '{label}'")
            else:
                print("No hand detected — sample skipped.")

        elif key == ord("n"):
            label = input("Enter new label: ").strip()
            count = label_counts.get(label, 0)

        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

        elif key == ord("c"):
            removed = clear_label(label)
            count = 0
            label_counts = get_label_counts()
            print(f"Cleared {removed} sample(s) for '{label}'. Ready to recollect.")

        elif key == ord("r"):
            target = input("Enter label name to reset/clear: ").strip()
            removed = clear_label(target)
            label_counts = get_label_counts()
            if target == label:
                count = 0
            print(f"Cleared {removed} sample(s) for '{target}'.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Data saved to {DATA_FILE}")


if __name__ == "__main__":
    main()
