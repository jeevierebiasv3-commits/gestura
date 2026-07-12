"""
autocollect_sequences.py  (Stage 2 — hands-free bulk recording)
---------------------------------------------------------------
Same output as collect_sequences.py (sequences/<LABEL>/<index>.npy of raw
126-vectors), but you don't press SPACE for every clip. Two hands-free modes let
you record MANY clips of a sign in one go, then move to the next sign:

  MOTION mode (default): reuses the live app's motion-gated segmentation. Just
    perform the sign; when you start moving it records, when you settle it saves
    the clip automatically and re-arms for the next repetition. Sign, pause,
    sign, pause — clips pile up on their own.

  TIMER mode (--timer): records fixed-length clips on a countdown. A 3-2-1
    counts you in, records for CLIP_SECONDS, saves, rests, repeats. Good for
    signs where you'd rather not rely on motion detection.

Stop early once you hit the target count per label. Everything else (format,
folders, appending) is identical to collect_sequences.py so
train_sequence_model.py consumes it unchanged.

USAGE
    python autocollect_sequences.py                    # motion mode, prompts label
    python autocollect_sequences.py --label HELLO      # skip the prompt
    python autocollect_sequences.py --target 30        # auto-advance hint at 30
    python autocollect_sequences.py --timer            # fixed-length timer mode

Controls (window): n = new label | q = quit | SPACE = pause/resume |
                   z = undo (cancel current clip, else delete last saved clip) |
                   x = clear ALL clips for this label (press twice to confirm)
"""

import os
import time
import argparse

import cv2
import numpy as np
from collections import deque

import hand_features as hf
import live_sequence as ls   # reuse the exact motion thresholds + segmentation

SEQ_DIR = "sequences"
MIN_FRAMES = 8

# timer-mode settings (all overridable from the command line — see main())
CLIP_SECONDS = 3.0     # how long each fixed clip records (raised from 2.0 so
                       # slower/longer signs finish inside the window)
COUNTIN_SECONDS = 2.0  # 3-2-1 style lead-in before each clip
REST_SECONDS = 0.8     # brief rest after saving before the next count-in


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


def _clip_paths(label):
    d = label_dir(label)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy")]


def delete_last_clip(label):
    """Delete the highest-numbered clip for this label; return its path (or None).

    Removing the NEWEST clip keeps the 0000, 0001, ... numbering contiguous, so
    the next save_clip reuses the freed index and no gaps appear."""
    paths = _clip_paths(label)
    if not paths:
        return None

    def stem_num(p):
        s = os.path.splitext(os.path.basename(p))[0]
        return int(s) if s.isdigit() else -1

    newest = max(paths, key=stem_num)
    os.remove(newest)
    return newest


def clear_label(label):
    """Delete ALL clips for this label. Returns how many were removed."""
    paths = _clip_paths(label)
    for p in paths:
        os.remove(p)
    return len(paths)


def outlined(frame, text, pos, scale, color, th=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def hud(frame, label, clips, target, mode, extra, extra_color, paused,
        notice="", notice_color=(255, 255, 255)):
    outlined(frame, f"Label: {label}   Clips: {clips}"
             + (f" / {target}" if target else ""), (10, 30), 0.8,
             (0, 255, 0) if not (target and clips >= target) else (0, 220, 255), 2)
    outlined(frame, f"[{mode}]  n=new  q=quit  SPACE=pause  z=undo  x=clear",
             (10, 58), 0.55, (220, 220, 220), 1)
    if paused:
        outlined(frame, "PAUSED (SPACE to resume)", (10, 86), 0.7, (0, 220, 255), 2)
    elif extra:
        outlined(frame, extra, (10, 86), 0.8, extra_color, 2)
    if notice:
        outlined(frame, notice, (10, 118), 0.6, notice_color, 2)


def handle_edit_key(key, label, clips, recording, clear_armed):
    """Shared z=undo / x=clear handling for both capture modes.

    Returns (clips, notice, notice_color, clear_armed, cancel_recording):
      - z: if a clip is being recorded RIGHT NOW, cancel it (drop the buffer,
        nothing saved). Otherwise delete the last SAVED clip from disk.
      - x: clear ALL clips for this label, but only on a SECOND x press
        (first press arms a confirmation so you can't wipe a label by accident).
      - any other key disarms a pending clear.
    `cancel_recording` tells the caller to drop its in-progress segment buffer."""
    notice, notice_color, cancel = "", (255, 255, 255), False

    if key == ord("z"):
        clear_armed = False
        if recording:
            cancel = True
            notice, notice_color = "canceled current clip (not saved)", (0, 220, 255)
        else:
            removed = delete_last_clip(label)
            if removed:
                clips -= 1
                notice = f"deleted last clip -> {clips} left"
                notice_color = (0, 180, 255)
            else:
                notice, notice_color = "nothing to undo", (200, 200, 200)
    elif key == ord("x"):
        if clear_armed:
            n = clear_label(label)
            clips = 0
            clear_armed = False
            notice, notice_color = f"cleared ALL {n} clips for '{label}'", (0, 0, 255)
        else:
            clear_armed = True
            notice = f"press x AGAIN to delete all {clips} clips (any other key cancels)"
            notice_color = (0, 0, 255)
    else:
        clear_armed = False   # any unrelated key disarms a pending clear

    return clips, notice, notice_color, clear_armed, cancel


def run_motion_mode(cap, window, label, target):
    """Auto-capture using live_sequence's motion-gated segmentation."""
    clips = count_clips(label)
    preroll = deque(maxlen=ls.PREROLL)
    segment = []
    collecting = False
    still_count = 0
    prev_raw = None
    paused = False
    notice, notice_color, notice_until, clear_armed = "", (255, 255, 255), 0.0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = ls.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                ls.mp_drawing.draw_landmarks(
                    frame, hlm, ls.mp.solutions.hands.HAND_CONNECTIONS,
                    ls.mp_styles.get_default_hand_landmarks_style(),
                    ls.mp_styles.get_default_hand_connections_style())

        raw = hf.extract_raw_hands(results)
        hand_present = any(v != 0.0 for v in raw)
        motion = hf.frame_motion(prev_raw, raw) if prev_raw is not None else 0.0

        extra, extra_color = "READY — sign now", ls.COLOR_HIGH
        if not paused:
            if not collecting:
                preroll.append(raw)
                if hand_present and motion >= ls.MOTION_START:
                    collecting = True
                    segment = list(preroll)
                    still_count = 0
            else:
                segment.append(raw)
                extra = f"RECORDING...  {len(segment)} frames"
                extra_color = ls.COLOR_CURRENT
                if motion < ls.MOTION_STOP or not hand_present:
                    still_count += 1
                else:
                    still_count = 0
                if still_count >= ls.STOP_FRAMES or len(segment) >= ls.MAX_SEG:
                    if len(segment) >= max(MIN_FRAMES, ls.MIN_SEG):
                        path = save_clip(label, segment)
                        clips += 1
                        print(f"Saved clip #{clips} for '{label}' "
                              f"({len(segment)} frames) -> {path}")
                    collecting = False
                    segment = []
                    preroll.clear()

        if collecting:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 12, (0, 0, 255), -1)
        shown_notice = notice if time.time() < notice_until else ""
        hud(frame, label, clips, target, "MOTION", extra, extra_color, paused,
            shown_notice, notice_color)
        cv2.imshow(window, frame)
        prev_raw = raw

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("n"):
            return "new"
        if key == ord(" "):
            paused = not paused
            collecting = False
            segment = []
        elif key in (ord("z"), ord("x")):
            clips, notice, notice_color, clear_armed, cancel = handle_edit_key(
                key, label, clips, collecting, clear_armed)
            notice_until = time.time() + 2.5
            if cancel:
                collecting = False
                segment = []
                still_count = 0
                preroll.clear()


def run_timer_mode(cap, window, label, target):
    """Auto-capture fixed-length clips on a repeating count-in / record cycle."""
    clips = count_clips(label)
    # phases: "countin" -> "record" -> "rest" -> back to countin
    phase = "countin"
    phase_end = time.time() + COUNTIN_SECONDS
    segment = []
    paused = False
    notice, notice_color, notice_until, clear_armed = "", (255, 255, 255), 0.0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = ls.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                ls.mp_drawing.draw_landmarks(
                    frame, hlm, ls.mp.solutions.hands.HAND_CONNECTIONS,
                    ls.mp_styles.get_default_hand_landmarks_style(),
                    ls.mp_styles.get_default_hand_connections_style())
        raw = hf.extract_raw_hands(results)

        now = time.time()
        extra, extra_color = "", ls.COLOR_HIGH
        if not paused:
            remaining = phase_end - now
            if phase == "countin":
                extra = f"GET READY... {max(0, remaining):.0f}"
                extra_color = (0, 220, 255)
                if remaining <= 0:
                    phase, phase_end, segment = "record", now + CLIP_SECONDS, []
            elif phase == "record":
                segment.append(raw)
                extra = f"RECORDING...  {remaining:.1f}s"
                extra_color = ls.COLOR_CURRENT
                cv2.circle(frame, (frame.shape[1] - 40, 40), 12, (0, 0, 255), -1)
                if remaining <= 0:
                    if len(segment) >= MIN_FRAMES:
                        path = save_clip(label, segment)
                        clips += 1
                        print(f"Saved clip #{clips} for '{label}' "
                              f"({len(segment)} frames) -> {path}")
                    phase, phase_end = "rest", now + REST_SECONDS
            else:  # rest
                extra = "rest..."
                if remaining <= 0:
                    phase, phase_end = "countin", now + COUNTIN_SECONDS

        shown_notice = notice if now < notice_until else ""
        hud(frame, label, clips, target, "TIMER", extra, extra_color, paused,
            shown_notice, notice_color)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("n"):
            return "new"
        if key == ord(" "):
            paused = not paused
            if not paused:  # restart cleanly from a fresh count-in
                phase, phase_end, segment = "countin", time.time() + COUNTIN_SECONDS, []
        elif key in (ord("z"), ord("x")):
            clips, notice, notice_color, clear_armed, cancel = handle_edit_key(
                key, label, clips, phase == "record", clear_armed)
            notice_until = time.time() + 2.5
            if cancel:  # abort the current recording, restart from count-in
                phase, phase_end, segment = "countin", time.time() + COUNTIN_SECONDS, []


def main():
    # timer-length flags below override these module globals; run_timer_mode
    # reads them, so updating here reconfigures the timer
    global CLIP_SECONDS, COUNTIN_SECONDS
    ap = argparse.ArgumentParser(description="Hands-free bulk clip recording.")
    ap.add_argument("--label", default=None, help="label to record (skips prompt)")
    ap.add_argument("--target", type=int, default=0,
                    help="clip-count goal per label, shown in the HUD (0 = none)")
    ap.add_argument("--timer", action="store_true",
                    help="fixed-length timer mode instead of motion mode")
    ap.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS,
                    help=f"timer mode: seconds to record each clip "
                         f"(default {CLIP_SECONDS}; raise it if a sign gets cut off)")
    ap.add_argument("--countin", type=float, default=COUNTIN_SECONDS,
                    help=f"timer mode: count-in seconds before each clip "
                         f"(default {COUNTIN_SECONDS})")
    args = ap.parse_args()

    CLIP_SECONDS = args.clip_seconds
    COUNTIN_SECONDS = args.countin

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    window = "Auto-Collect Sequences"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    mode = "TIMER" if args.timer else "MOTION"
    print(f"[{mode} mode] Controls: n=new label  q=quit  SPACE=pause  "
          f"z=undo  x=clear-all(2x)\n")

    label = args.label or input("Enter label for this sign (e.g. HELLO): ").strip()
    runner = run_timer_mode if args.timer else run_motion_mode

    while True:
        result = runner(cap, window, label, args.target)
        if result == "quit":
            break
        if result == "new":
            label = input("Enter new label: ").strip()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Sequences saved under ./{SEQ_DIR}/  Next: python train_sequence_model.py")


if __name__ == "__main__":
    main()
