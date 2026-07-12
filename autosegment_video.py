"""
autosegment_video.py  (Stage 2 — auto-draft timestamps from one multi-sign video)
---------------------------------------------------------------------------------
You have ONE video of many signs back-to-back and don't want to note every
start/end time by hand. This scans the video with MediaPipe, finds each burst of
hand MOTION (the same motion-gated logic the live app uses), and writes a DRAFT
timestamps file with the windows pre-filled and placeholder labels:

    SIGN_01 , 0:04.1 , 0:07.3
    SIGN_02 , 0:08.0 , 0:11.6
    ...

Your ONLY manual step then is renaming SIGN_01 -> HELLO, SIGN_02 -> THANK_YOU,
etc. (a preview frame per window is saved under ./previews/ to help you tell them
apart). Then feed the file to split_video_by_timestamps.py to extract the clips.

USAGE
    python autosegment_video.py my_video.mp4
    python autosegment_video.py my_video.mp4 --out my_times.txt --no-previews

Tuning (if it finds too many / too few windows):
    --motion-start / --motion-stop   raise to ignore small movements
    --min-frames                     drop very short bursts (twitches)
This does NOT create training data — it only drafts the timestamps file. Review
it, fix the labels, then run split_video_by_timestamps.py.
"""

import os
import sys
import argparse

import cv2

import hand_features as hf
import live_sequence as ls   # reuse the live app's motion thresholds as defaults

PREVIEW_DIR = "previews"


def fmt_time(seconds):
    """seconds -> M:SS.s  (matches the timestamps-file format)."""
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"


def main():
    ap = argparse.ArgumentParser(
        description="Auto-draft a timestamps file from one multi-sign video.")
    ap.add_argument("video", help="path to the source video")
    ap.add_argument("--out", default="timestamps.txt",
                    help="draft file to write (default: timestamps.txt)")
    ap.add_argument("--motion-start", type=float, default=ls.MOTION_START,
                    help=f"motion to begin a window (default {ls.MOTION_START})")
    ap.add_argument("--motion-stop", type=float, default=ls.MOTION_STOP,
                    help=f"motion below this = settled (default {ls.MOTION_STOP})")
    ap.add_argument("--stop-frames", type=int, default=ls.STOP_FRAMES,
                    help=f"settled frames that end a window (default {ls.STOP_FRAMES})")
    ap.add_argument("--min-frames", type=int, default=ls.MIN_SEG,
                    help=f"drop windows shorter than this (default {ls.MIN_SEG})")
    ap.add_argument("--no-previews", action="store_true",
                    help="don't save a preview frame per window")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)
    if os.path.exists(args.out):
        print(f"{args.out} already exists — move/rename it first so a draft "
              f"doesn't overwrite your edits.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    hands = hf.make_hands(static_image_mode=False)

    segments = []          # list of (start_frame, end_frame)
    collecting = False
    seg_start = 0
    still_count = 0
    prev_raw = None
    frame_idx = 0

    # remember one middle frame per finished window for the preview thumbnails
    want_previews = not args.no_previews
    mid_frames = []        # (segment_index, frame_index_to_grab)

    print(f"Scanning {args.video} at {fps:.0f} fps for sign windows...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        raw = hf.extract_raw_hands(results)
        hand_present = any(v != 0.0 for v in raw)
        motion = hf.frame_motion(prev_raw, raw) if prev_raw is not None else 0.0

        if not collecting:
            if hand_present and motion >= args.motion_start:
                collecting = True
                seg_start = frame_idx
                still_count = 0
        else:
            if motion < args.motion_stop or not hand_present:
                still_count += 1
            else:
                still_count = 0
            if still_count >= args.stop_frames:
                end = frame_idx - args.stop_frames   # trim the trailing still tail
                if end - seg_start >= args.min_frames:
                    segments.append((seg_start, end))
                    mid_frames.append((len(segments) - 1, (seg_start + end) // 2))
                collecting = False

        prev_raw = raw
        frame_idx += 1

    # flush a window still open at end-of-video
    if collecting and (frame_idx - seg_start) >= args.min_frames:
        segments.append((seg_start, frame_idx))
        mid_frames.append((len(segments) - 1, (seg_start + frame_idx) // 2))

    if not segments:
        print("No motion windows found. The video may have low hand-detection, "
              "or try lowering --motion-start.")
        cap.release()
        sys.exit(1)

    # write the draft timestamps file
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# DRAFT — rename each SIGN_## to the real sign, then run:\n")
        f.write(f"#   python split_video_by_timestamps.py {args.video}\n")
        f.write("# LABEL , START , END\n\n")
        for i, (s, e) in enumerate(segments, 1):
            f.write(f"SIGN_{i:02d} , {fmt_time(s / fps)} , {fmt_time(e / fps)}\n")

    print(f"\nFound {len(segments)} sign window(s). Draft written to {args.out}")

    # save preview thumbnails (a second pass to grab specific frames)
    if want_previews:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        wanted = {fi: si for si, fi in mid_frames}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        grabbed = 0
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                si = wanted.pop(idx)
                out = os.path.join(PREVIEW_DIR, f"SIGN_{si + 1:02d}.jpg")
                cv2.imwrite(out, frame)
                grabbed += 1
            idx += 1
        print(f"Saved {grabbed} preview frame(s) to ./{PREVIEW_DIR}/ "
              f"(open them to identify each SIGN_##).")

    cap.release()
    print("\nNext:")
    print(f"  1. Open {args.out}, rename SIGN_## to real labels (check ./{PREVIEW_DIR}/).")
    print(f"  2. python split_video_by_timestamps.py {args.video}")


if __name__ == "__main__":
    main()
