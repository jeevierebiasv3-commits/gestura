"""
extract_from_videos.py  (Stage 2 — bulk data from videos, no live recording)
----------------------------------------------------------------------------
Turn a folder of sign VIDEOS into training clips automatically, so you don't
have to perform every sign in front of the webcam. Feeds MediaPipe the same way
collect_sequences.py does and writes the SAME output format
(sequences/<LABEL>/<index>.npy of raw 126-vectors), so train_sequence_model.py
picks them up with zero changes.

INPUT LAYOUT — one subfolder per sign label, videos inside:
    videos/
      HELLO/       clip1.mp4  clip2.mp4 ...
      THANK_YOU/   a.mov      b.mp4 ...
      NONE/        random_nonsigns.mp4 ...
Each video file becomes ONE clip (one .npy). This matches how datasets like
WLASL / MS-ASL are organized (one instance = one video of one sign).

USAGE
    python extract_from_videos.py                 # process ./videos -> ./sequences
    python extract_from_videos.py --src my_vids   # custom input folder
    python extract_from_videos.py --label HELLO   # only that one label

Notes
- Leading/trailing frames with NO hand detected are trimmed (dataset videos
  often have a still intro/outro). Interior no-hand frames are kept as zeros so
  motion/timing is preserved.
- A video where hands are almost never found is skipped and reported, so you can
  see which signs need a better source (or a self-recorded clip instead).
- Re-running APPENDS (new indices), it does not overwrite — safe to run again
  after adding more videos.
"""

import os
import sys
import argparse

import cv2
import numpy as np

import hand_features as hf

SEQ_DIR = "sequences"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

MIN_FRAMES = 8         # discard clips shorter than this (same as live collector)
MIN_HAND_RATIO = 0.15  # skip a video if hands are found in < this fraction of frames


def label_dir(label):
    return os.path.join(SEQ_DIR, label)


def count_clips(label):
    d = label_dir(label)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def save_clip(label, frames):
    d = label_dir(label)
    os.makedirs(d, exist_ok=True)
    idx = count_clips(label)
    path = os.path.join(d, f"{idx:04d}.npy")
    np.save(path, np.asarray(frames, dtype="float32"))
    return path


def trim_empty_ends(frames):
    """Drop leading/trailing frames that have no hand (all zeros). Keeps interior
    empties so motion timing between hand appearances is preserved."""
    def has_hand(f):
        return any(v != 0.0 for v in f)

    start = 0
    while start < len(frames) and not has_hand(frames[start]):
        start += 1
    end = len(frames)
    while end > start and not has_hand(frames[end - 1]):
        end -= 1
    return frames[start:end]


def process_video(path, hands):
    """Return (clip_frames, hand_ratio). clip_frames is a list of raw 126-vectors
    with empty ends trimmed; hand_ratio is fraction of frames a hand was found."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0

    frames = []
    hand_hits = 0
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        raw = hf.extract_raw_hands(results)
        if any(v != 0.0 for v in raw):
            hand_hits += 1
        frames.append(raw)
    cap.release()

    ratio = (hand_hits / total) if total else 0.0
    return trim_empty_ends(frames), ratio


def main():
    ap = argparse.ArgumentParser(description="Extract training clips from videos.")
    ap.add_argument("--src", default="videos",
                    help="input folder of <LABEL>/ subfolders (default: videos)")
    ap.add_argument("--label", default=None,
                    help="only process this one label subfolder")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"No input folder ./{args.src}/. Create it with one subfolder per "
              f"sign, e.g. {args.src}/HELLO/clip1.mp4")
        sys.exit(1)

    labels = sorted(d for d in os.listdir(args.src)
                    if os.path.isdir(os.path.join(args.src, d)))
    if args.label:
        labels = [l for l in labels if l == args.label]
        if not labels:
            print(f"Label '{args.label}' not found under ./{args.src}/")
            sys.exit(1)
    if not labels:
        print(f"No label subfolders inside ./{args.src}/. Expected e.g. "
              f"{args.src}/HELLO/, {args.src}/THANK_YOU/")
        sys.exit(1)

    # static_image_mode=False: treat each video as a continuous stream (tracking
    # across frames), which is what we want for motion signs.
    hands = hf.make_hands(static_image_mode=False)

    grand_saved = 0
    grand_skipped = 0
    for label in labels:
        vids = sorted(f for f in os.listdir(os.path.join(args.src, label))
                      if f.lower().endswith(VIDEO_EXTS))
        if not vids:
            print(f"[{label}] no video files — skipped")
            continue

        print(f"\n[{label}] {len(vids)} video(s)")
        saved = skipped = 0
        for v in vids:
            path = os.path.join(args.src, label, v)
            clip, ratio = process_video(path, hands)
            if clip is None:
                print(f"  ! could not open {v}")
                skipped += 1
                continue
            if len(clip) < MIN_FRAMES:
                print(f"  - {v}: only {len(clip)} usable frames — skipped")
                skipped += 1
                continue
            if ratio < MIN_HAND_RATIO:
                print(f"  - {v}: hands found in only {ratio*100:.0f}% of frames "
                      f"— skipped (bad source for MediaPipe)")
                skipped += 1
                continue
            out = save_clip(label, clip)
            saved += 1
            print(f"  + {v}: {len(clip)} frames (hands {ratio*100:.0f}%) -> {out}")

        print(f"  = {label}: saved {saved}, skipped {skipped} "
              f"(total clips now: {count_clips(label)})")
        grand_saved += saved
        grand_skipped += skipped

    print(f"\nDone. Saved {grand_saved} clip(s), skipped {grand_skipped}, "
          f"under ./{SEQ_DIR}/")
    print("Next: python train_sequence_model.py")


if __name__ == "__main__":
    main()
