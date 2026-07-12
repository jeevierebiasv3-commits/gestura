"""
split_video_by_timestamps.py  (Stage 2 — carve one multi-sign video into clips)
-------------------------------------------------------------------------------
You have ONE video that contains many signs back-to-back (e.g. a "25 basic ASL
signs" tutorial). This reads a small timestamps file you write (one line per
sign: label, start, end) and extracts each time-window straight into the trainable
format sequences/<LABEL>/<index>.npy — the SAME output as collect_sequences.py,
so train_sequence_model.py consumes it unchanged. No ffmpeg required (pure OpenCV
+ MediaPipe), so a matching sign appears as landmark frames, not a cut video file.

The video is read ONCE start-to-finish; each frame is dropped into whatever
timestamp window it falls in. One line -> one clip. Re-running APPENDS (safe).

TIMESTAMPS FILE  (default: timestamps.txt) — one sign per line:
    # anything after a '#' is a comment; blank lines ignored
    # LABEL , START , END        times as SS, MM:SS, or MM:SS.mmm
    HELLO      , 0:04 , 0:07
    THANK_YOU  , 0:08 , 0:12
    YES        , 0:13 , 0:15
    NONE       , 0:40 , 0:45     # idle / non-sign footage is great as a NONE class

USAGE
    python split_video_by_timestamps.py my_video.mp4
    python split_video_by_timestamps.py my_video.mp4 --times my_times.txt
    python split_video_by_timestamps.py --make-template     # write a sample file

TIP: it's ONE example per sign — a starting seed. A sequence model needs >= 2
clips per label to even split train/test, and ~15-30 to learn well. After this,
bulk up each sign with:  python autocollect_sequences.py --label HELLO --target 30
"""

import os
import sys
import argparse

import cv2
import numpy as np

import hand_features as hf

SEQ_DIR = "sequences"
MIN_FRAMES = 8      # windows yielding fewer usable frames than this are skipped
TEMPLATE = "timestamps.txt"

_SAMPLE = """# One line per sign:  LABEL , START , END
# Times may be SS (7), MM:SS (0:07), or MM:SS.mmm (0:07.5).
# Lines starting with '#' and blank lines are ignored.
# Use underscores (no spaces) in labels. Add a NONE window of idle/non-sign
# footage so the model has an explicit "not a sign" bucket.

HELLO      , 0:04 , 0:07
THANK_YOU  , 0:08 , 0:12
YES        , 0:13 , 0:15
NO         , 0:16 , 0:19
# NONE     , 0:40 , 0:45
"""


def parse_time(s):
    """'7' | '0:07' | '1:05.5'  -> seconds (float). Raises ValueError if bad."""
    s = s.strip()
    if ":" in s:
        mm, ss = s.rsplit(":", 1)
        return int(mm) * 60 + float(ss)
    return float(s)


def load_timestamps(path):
    """Parse the timestamps file -> list of (label, start_s, end_s). Reports and
    skips malformed lines instead of crashing."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                print(f"  ! line {lineno}: expected 'LABEL, START, END' -> skipped")
                continue
            label, start_s, end_s = parts
            try:
                start, end = parse_time(start_s), parse_time(end_s)
            except ValueError:
                print(f"  ! line {lineno}: bad time value -> skipped")
                continue
            if end <= start:
                print(f"  ! line {lineno}: end <= start -> skipped")
                continue
            rows.append((label, start, end))
    return rows


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


def main():
    ap = argparse.ArgumentParser(
        description="Split one multi-sign video into per-sign clips by timestamps.")
    ap.add_argument("video", nargs="?", help="path to the source video")
    ap.add_argument("--times", default=TEMPLATE,
                    help=f"timestamps file (default: {TEMPLATE})")
    ap.add_argument("--make-template", action="store_true",
                    help="write a sample timestamps file and exit")
    args = ap.parse_args()

    if args.make_template:
        if os.path.exists(TEMPLATE):
            print(f"{TEMPLATE} already exists — not overwriting.")
        else:
            with open(TEMPLATE, "w", encoding="utf-8") as f:
                f.write(_SAMPLE)
            print(f"Wrote {TEMPLATE}. Fill in your signs/times, then run:\n"
                  f"  python split_video_by_timestamps.py YOUR_VIDEO.mp4")
        return

    if not args.video:
        print("Give the video path, e.g.:\n"
              "  python split_video_by_timestamps.py my_video.mp4\n"
              "Need the timestamps file? Run:  "
              "python split_video_by_timestamps.py --make-template")
        sys.exit(1)
    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.times):
        print(f"Timestamps file not found: {args.times}\n"
              f"Create one with:  python split_video_by_timestamps.py --make-template")
        sys.exit(1)

    print(f"Reading timestamps from {args.times} ...")
    windows = load_timestamps(args.times)
    if not windows:
        print("No valid timestamp lines found. Nothing to do.")
        sys.exit(1)
    print(f"{len(windows)} sign window(s) to extract.\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    hands = hf.make_hands(static_image_mode=False)
    buffers = [[] for _ in windows]   # collected raw frames per window

    # single pass: bucket each frame into every window whose [start,end) it hits
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        active = [i for i, (_, s, e) in enumerate(windows) if s <= t < e]
        if not active:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        raw = hf.extract_raw_hands(results)
        for i in active:
            buffers[i].append(raw)
    cap.release()

    saved = skipped = 0
    for (label, s, e), buf in zip(windows, buffers):
        usable = [f for f in buf if any(v != 0.0 for v in f)]
        if len(buf) < MIN_FRAMES:
            print(f"  - {label} [{s:.1f}-{e:.1f}s]: only {len(buf)} frames "
                  f"— skipped (window too short or past end of video)")
            skipped += 1
            continue
        if len(usable) < MIN_FRAMES:
            print(f"  - {label} [{s:.1f}-{e:.1f}s]: hands seen in only "
                  f"{len(usable)} frames — skipped (MediaPipe found few hands)")
            skipped += 1
            continue
        path = save_clip(label, buf)
        saved += 1
        print(f"  + {label}: {len(buf)} frames "
              f"(hands in {len(usable)}) -> {path}")

    print(f"\nExtracted {saved} clip(s), skipped {skipped}, under ./{SEQ_DIR}/")
    if saved:
        print("\nThis is ONE clip per sign — a seed, not enough to train well.")
        print("Bulk each sign up to ~15-30 clips, then train:")
        print("  python autocollect_sequences.py --label HELLO --target 30")
        print("  python train_sequence_model.py")


if __name__ == "__main__":
    main()
