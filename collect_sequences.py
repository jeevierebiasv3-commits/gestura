"""
collect_sequences.py  (Stage 2 — motion signs)
----------------------------------------------
Records short CLIPS of hand-landmark frames so a sequence model (GRU/LSTM) can
learn MOTION-based signs (wave for "Hello", hand-from-chin for "Thank you",
finger-wag for "No", etc.) that a single static frame cannot represent.

Clips are VARIABLE LENGTH: you press SPACE to start recording, perform the sign
at its natural speed (short or long), then press SPACE again to stop. This lets
you capture long / multi-part signs, not just fixed ~1.5s ones. Each clip is a
sequence of raw 126-number landmark vectors, saved as a .npy array under:
    sequences/<LABEL>/<index>.npy
Raw (un-normalized) landmarks are stored so training can normalize/augment and
resample every clip to a fixed length (hf.SEQ_LEN) — same philosophy as the
static data.csv.

HOW TO USE
1. Run: python collect_sequences.py
2. Type a label (e.g. HELLO).
3. Press SPACE to START recording, perform the sign, press SPACE again to STOP.
   Repeat ~20-40 times per sign, varying speed/position a little. For a long
   sign, just take as long as you need before pressing SPACE to stop.
4. Press 'n' for a new label, 'q' to quit.

Tip: also record a NONE / Idle label (still hand, hand facing away, random
poses, moving between signs) so the model has an explicit "not a sign" bucket.
"""

import os
import cv2
import numpy as np

import mediapipe as mp
import hand_features as hf

SEQ_DIR = "sequences"
MIN_FRAMES = 8       # clips shorter than this are discarded (accidental taps)
MAX_FRAMES = 180     # hard cap (~9s) so a forgotten SPACE can't record forever

# Chars Windows forbids in file/folder names. A label like "How are you?"
# can't be a directory, so we sanitize the FOLDER name but keep the TRUE
# label (with ? or !) in a sidecar label.txt that training reads back —
# so punctuation survives into seq_labels.json and the spoken sentence.
_ILLEGAL = '<>:"/\\|?*'


def safe_dirname(label):
    name = "".join("_" if ch in _ILLEGAL else ch for ch in label)
    name = name.rstrip(" .")   # Windows also dislikes trailing space/dot
    return name or "_"

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = hf.make_hands()


def label_dir(label):
    return os.path.join(SEQ_DIR, safe_dirname(label))


def count_clips(label):
    d = label_dir(label)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def save_clip(label, frames):
    d = label_dir(label)
    os.makedirs(d, exist_ok=True)
    # remember the true label (with any ? or ! the folder name had to drop)
    sidecar = os.path.join(d, "label.txt")
    if not os.path.isfile(sidecar):
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(label)
    idx = count_clips(label)
    path = os.path.join(d, f"{idx:04d}.npy")
    np.save(path, np.asarray(frames, dtype="float32"))
    return path


def outlined(frame, text, pos, scale, color, th=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    label = input("Enter label for this motion sign (e.g. HELLO): ").strip()
    clips = count_clips(label)

    WINDOW = "Collect Motion Sequences"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    print("\nControls: SPACE=start/stop recording | n=new label | q=quit\n")

    recording = False
    buffer = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        h, w = frame.shape[:2]

        if recording:
            buffer.append(hf.extract_raw_hands(results))
            # red dot + live frame count; auto-stop at the hard cap
            cv2.circle(frame, (w - 40, 40), 12, (0, 0, 255), -1)
            outlined(frame, f"REC  {len(buffer)} frames  (SPACE=stop)",
                     (10, 30), 0.8, (0, 0, 255), 2)
            if len(buffer) >= MAX_FRAMES:
                path = save_clip(label, buffer)
                clips += 1
                print(f"Hit {MAX_FRAMES}-frame cap. Saved clip #{clips} "
                      f"for '{label}' ({len(buffer)} frames) -> {path}")
                buffer = []
                recording = False
        else:
            outlined(frame, f"Label: {label}  Clips: {clips}", (10, 30), 0.8, (0, 255, 0), 2)
            outlined(frame, "SPACE=start recording  n=new label  q=quit",
                     (10, 60), 0.55, (220, 220, 220), 1)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording = True
                buffer = []
                print("Recording... perform the sign, press SPACE again to stop.")
            else:
                recording = False
                if len(buffer) >= MIN_FRAMES:
                    path = save_clip(label, buffer)
                    clips += 1
                    print(f"Saved clip #{clips} for '{label}' "
                          f"({len(buffer)} frames) -> {path}")
                else:
                    print(f"Clip too short ({len(buffer)} < {MIN_FRAMES} frames) "
                          f"— discarded.")
                buffer = []
        elif key == ord("n") and not recording:
            label = input("Enter new label: ").strip()
            clips = count_clips(label)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Sequences saved under ./{SEQ_DIR}/")


if __name__ == "__main__":
    main()
