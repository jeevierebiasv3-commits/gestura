"""
live_sequence.py  (Stage 2 — motion signs, with segmentation)
-------------------------------------------------------------
Live demo for the GRU motion model (seq_model.h5). Instead of a fixed-length
sliding window, this uses MOTION-GATED SEGMENTATION:

  - watch frame-to-frame motion (hf.frame_motion)
  - when the hand starts moving, begin buffering a "sign segment"
  - keep buffering for as long as the sign lasts (any length)
  - when the hand settles (motion low for a short while), the sign has ended:
    resample that whole segment to SEQ_LEN, normalize, and classify it

This handles LONG / variable-length signs (the segment is however long the sign
takes) and, because it returns to "listening" after each sign, it also
translates several signs performed back-to-back with a brief pause between them
(pause-separated continuous signing) — the recognized signs are appended into a
running sentence.

Run: python live_sequence.py   (needs seq_model.h5 + seq_labels.json)
Controls: 'c' = clear sentence | 't' = toggle voice | 'q' = quit
The sentence auto-clears after a few idle seconds, is capped to the most recent
signs, colors each sign by confidence (green=high, yellow=borderline), and shows
a cyan "..." while a sign is being recorded. Each recognized sign is also spoken
aloud (offline text-to-speech, toggle with 't').
"""

import cv2
import json
import time
import queue
import threading
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

import mediapipe as mp
import hand_features as hf
import suggest              # shared bigram model (autocomplete + context prior)

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

MODEL_FILE = "seq_model.h5"
LABELS_FILE = "seq_labels.json"
SEQ_LEN = hf.SEQ_LEN

# --- classification gates (reject unsure / ambiguous segments) ---
# Relaxed from 0.85/0.30 to translate more often while the dataset is still
# small/imbalanced. Raise them again once you have ~30-50 balanced clips/class
# (fewer wrong guesses); lower further if real signs are still being missed.
CONFIDENCE_THRESHOLD = 0.85   # top class must be at least this confident
MARGIN_THRESHOLD = 0.30       # ...AND beat the 2nd-best class by this much

# --- context prior (re-rank softmax by the previously committed sign) ---
# How much the learned bigram prior (suggest.transition_prior) is blended into
# the raw softmax: probs = probs*(1-a) + a*prior. Kept SMALL on purpose — the
# recognizer stays in charge; the prior only nudges borderline near-ties toward a
# grammatically-likely continuation. Verified that at 0.05 a genuinely confident
# sign (>=0.90) stays above the 0.85 gate even when it's contextually unlikely,
# so the prior can't silently turn an accepted clear sign into a rejection. Raise
# it for a stronger grammar effect (at the cost of eroding that headroom); 0
# disables the prior entirely (pure recognizer).
CONTEXT_ALPHA = 0.05

# --- motion-gated segmentation (tune to your camera / signing speed) ---
MOTION_START = 0.006   # per-frame motion at/above this counts toward starting a
                       # sign. Lowered from 0.012: gentle, non-shaky motion signs
                       # (Hello = a soft wave, Well done) hover around 0.008-0.010
                       # and never crossed 0.012, so the segment wouldn't trigger.
                       # 0.006 sits at the top edge of still-hand jitter
                       # (~0.001-0.005), so START leans on START_FRAMES to avoid
                       # triggering on jitter. If a truly still hand starts false
                       # segments, your jitter is higher - raise this back toward
                       # 0.010 (watch the on-screen motion readout to pick a value).
START_FRAMES = 2       # need this many CONSECUTIVE frames at/above MOTION_START to
                       # begin (rejects single-frame jitter spikes without needing
                       # a high threshold). ~2 frames is still near-instant.
MOTION_STOP = 0.004    # ...and motion below this counts as "settled". Kept below
                       # MOTION_START (hysteresis) and above the ~0.005 jitter
                       # ceiling so a settled hand still ends the segment. Lowered
                       # from 0.008 alongside MOTION_START to keep the band's shape.
STOP_FRAMES = 14       # this many consecutive settled frames ends the segment.
                       # Raised from 8 (~0.27s) to ~0.5s: many signs here are
                       # TWO-BEAT (Good Morning, Thank you, See you later) with a
                       # brief still moment BETWEEN the beats. At 8 that interior
                       # pause ended the sign early, so only the first half was
                       # classified (missed or wrong). Being patient keeps the
                       # whole sign in one segment; the extra settling frames this
                       # adds to the tail are stripped again by trim_still_edges()
                       # before classification, so nothing is diluted. Cost: to
                       # separate two DISTINCT back-to-back signs you now pause a
                       # little longer (~0.5s) between them.
MIN_SEG = 10           # ignore segments shorter than this (twitches)
MAX_SEG = 180          # safety cap (~9s) on a single segment
PREROLL = 5            # frames kept from just before motion started
MIN_PRESENT_FRAC = 0.5 # a segment must have a hand VISIBLE in at least this
                       # fraction of its frames to be classified. MediaPipe drops
                       # tracking on a fast start / motion blur / a hand entering
                       # frame, leaving all-zero ("empty") frames. Trimming only
                       # strips still EDGES, so empty frames mid-sign survive and
                       # a mostly-empty segment would otherwise be classified as
                       # zeros -> a garbage word or a misleading reject reason.
                       # Below this fraction we reject with reason "lost tracking".

# --- sentence display ---
AUTO_CLEAR_SECONDS = 8.0    # clear the sentence after this many idle seconds
REPEAT_GAP_SECONDS = 2.5    # same sign re-appears if this long since the last add
                            # (raised from 1.0: one physical sign can split into
                            # two segments — a slow mid-sign moment or the hand
                            # drop after finishing — and each would commit the
                            # same word; a wider window collapses those into one.
                            # A deliberate re-sign after a real pause still counts.)
MAX_SENTENCE = 12          # keep only the most recent N signs
HIGH_CONF = 0.90           # >= this shows green; between gate and this = yellow

# colors are BGR (OpenCV)
COLOR_HIGH = (0, 220, 0)       # green  — confident sign
COLOR_BORDERLINE = (0, 220, 255)  # yellow — accepted but borderline
COLOR_CURRENT = (255, 255, 0)  # cyan   — sign currently being recorded

# labels that mean "not a real sign" — recognized but never displayed/appended
IGNORE_LABELS = {"none", "idle", "nothing", "background", "rest",
                 "not a hand sign", "this isn't a hand sign", "unknown"}

# --- text-to-speech ---
TTS_ENABLED = True    # speak each recognized sign aloud (toggle with 't')
TTS_RATE = 165        # words per minute (pyttsx3 default ~200)


class Speaker:
    """Speaks recognized signs aloud on a background thread.

    pyttsx3's runAndWait() blocks, so it runs in its own worker thread fed by a
    queue — the video loop never stalls waiting for speech. If pyttsx3 isn't
    installed the Speaker becomes a no-op (available == False) so the app still
    runs. Duplicate back-to-back words are skipped so a held sign isn't repeated.
    """

    def __init__(self, rate=TTS_RATE, enabled=True):
        self.available = _TTS_AVAILABLE
        self.enabled = enabled and _TTS_AVAILABLE
        self._q = queue.Queue()
        self._last = None
        self._thread = None
        if self.available:
            self._thread = threading.Thread(
                target=self._run, args=(rate,), daemon=True)
            self._thread.start()

    def _run(self, rate):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
        except Exception as e:
            print(f"[TTS] disabled — could not start speech engine: {e}")
            self.available = False
            self.enabled = False
            return
        while True:
            text = self._q.get()
            if text is None:          # shutdown sentinel
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass                  # never let a speech hiccup crash the app

    def say(self, text):
        """Queue a phrase to speak (dropped silently if TTS is off/unavailable)."""
        if not self.enabled or not text:
            return
        if text == self._last:        # don't repeat the immediately previous word
            return
        self._last = text
        self._q.put(text)

    def toggle(self):
        """Turn speech on/off at runtime. Returns the new state."""
        if not self.available:
            return False
        self.enabled = not self.enabled
        self._last = None
        return self.enabled

    def close(self):
        if self._thread is not None:
            self._q.put(None)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = hf.make_hands()


def conf_color(conf):
    """Green for a confident sign, yellow for a borderline (but accepted) one."""
    return COLOR_HIGH if conf >= HIGH_CONF else COLOR_BORDERLINE


def draw_subtitle(frame, tokens, max_lines=2, scale=1.0, th=2):
    """Draw a bottom subtitle bar from a list of (text, color) tokens.

    Tokens are word-wrapped to the frame width; only the most recent `max_lines`
    lines are shown (auto-scroll). Each token keeps its own color, so signs can
    be tinted by confidence and the in-progress marker shown distinctly."""
    if not tokens:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 24
    sep = 28                      # pixels of space between tokens
    max_w = w - 2 * margin

    def tok_w(t):
        (tw, _), _ = cv2.getTextSize(t, font, scale, th)
        return tw

    # greedy wrap into lines of tokens
    lines, cur, cur_w = [], [], 0
    for text, color in tokens:
        tw = tok_w(text)
        add = tw if not cur else tw + sep
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [], 0
            add = tw
        cur.append((text, color, tw))
        cur_w += add
    if cur:
        lines.append(cur)
    lines = lines[-max_lines:]

    (_, line_h), base = cv2.getTextSize("Hg", font, scale, th)
    step = line_h + base + 14
    bar = step * len(lines) + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = h - bar + line_h + 14
    for ln in lines:
        total = sum(tw for _, _, tw in ln) + sep * (len(ln) - 1)
        x = max(margin, (w - total) // 2)
        for text, color, tw in ln:
            cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, scale, color, th, cv2.LINE_AA)
            x += tw + sep
        y += step


def outlined(frame, text, pos, scale, color, th=2):
    x, y = pos
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def apply_context_prior(probs, prev_label, label_map=None):
    """Re-rank the raw softmax `probs` using the previously committed sign.

    Blends a learned bigram prior (suggest.transition_prior) into the softmax so
    grammatically-likely continuations are favored and unlikely ones suppressed:

        probs = probs * (1 - CONTEXT_ALPHA) + CONTEXT_ALPHA * prior

    Both distributions sum to 1, so the result does too (no renormalization) and
    a confident prediction is only mildly moved. It is a strict no-op when:
      - CONTEXT_ALPHA is 0,
      - there is no previous sign (prev_label falsy) — e.g. the first sign, or
      - the model has no transition data for prev_label (prior is None).
    `label_map` ({str(idx): label}) maps probs indices to labels for the prior.
    Best-effort: any failure returns `probs` unchanged."""
    if not prev_label or CONTEXT_ALPHA <= 0 or label_map is None:
        return probs
    try:
        labels = [label_map[str(i)] for i in range(len(probs))]
        prior = suggest.transition_prior(prev_label, labels)
        if prior is None:
            return probs
        prior = np.asarray(prior, dtype="float32")
        blended = probs * (1.0 - CONTEXT_ALPHA) + CONTEXT_ALPHA * prior
        s = float(blended.sum())
        return blended / s if s > 0 else probs
    except Exception:
        return probs


def trim_still_edges(segment, motion_stop=MOTION_STOP, keep=2):
    """Strip leading/trailing still frames from a raw segment before classifying.

    WHY: a live segment is padded on both ends with dead air the training clips
    never had — PREROLL idle frames captured before motion began, plus the
    STOP_FRAMES "settling" frames the segmenter keeps appending while it waits to
    confirm the sign ended. collect_sequences.py records tight SPACE-to-SPACE
    clips with none of that. After resample_sequence() squeezes a padded live
    clip to SEQ_LEN, the actual sign occupies fewer of the 45 frames and the
    motion channels get diluted -> lower confidence / wrong class (train/serve
    skew). Trimming the still edges makes the live clip look like a training clip.

    A frame is "still" if its motion vs. the previous frame is below motion_stop.
    We drop the still run at each end but keep `keep` frames of lead-in/-out so a
    sign that starts or ends gently isn't clipped into. Never trims below MIN_SEG
    frames; returns the segment unchanged if it can't trim safely."""
    seg = list(segment)
    n = len(seg)
    if n <= MIN_SEG:
        return seg
    # per-frame motion (frame 0 has no predecessor -> treat as still)
    mot = [0.0] + [hf.frame_motion(seg[i - 1], seg[i]) for i in range(1, n)]
    lo = 0
    while lo < n and mot[lo] < motion_stop:
        lo += 1
    hi = n - 1
    while hi > lo and mot[hi] < motion_stop:
        hi -= 1
    lo = max(0, lo - keep)
    hi = min(n - 1, hi + keep)
    if hi - lo + 1 < MIN_SEG:      # trimmed too aggressively — keep original
        return seg
    return seg[lo:hi + 1]


def classify_segment_debug(model, label_map, segment, prev_label=None):
    """Classify a segment and return FULL diagnostics (even when rejected).

    Returns a dict:
        top      : the model's best-guess label (always, even if rejected)
        conf     : confidence of the top class (0-1)
        second   : confidence of the runner-up (0-1)
        margin   : conf - second
        accepted : True if it cleared both gates (would be shown/appended)
        reason   : why it was rejected ("" when accepted)
    Use this to see WHY a sign didn't translate. `classify_segment` wraps it."""
    segment = trim_still_edges(segment)   # drop idle lead-in / settling tail

    # Reject a segment MediaPipe barely tracked: if the hand was visible in too
    # few of these frames, the rest are all-zero ("empty") frames and classifying
    # them yields a meaningless prediction. Surface it as "lost tracking" rather
    # than letting an empty clip masquerade as a low-confidence real sign.
    present = [any(v != 0.0 for v in f) for f in segment]
    present_frac = (sum(present) / len(present)) if present else 0.0
    if present_frac < MIN_PRESENT_FRAC:
        return {"top": "", "conf": 0.0, "second": 0.0, "margin": 0.0,
                "accepted": False, "reason": "lost tracking"}

    fixed = hf.resample_sequence(np.asarray(segment, dtype="float32"), SEQ_LEN)
    seq = hf.seq_features(fixed).reshape(1, SEQ_LEN, hf.SEQ_FEATURE_DIM)
    probs = model.predict(seq, verbose=0)[0]
    probs = apply_context_prior(probs, prev_label, label_map)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    second = float(np.partition(probs, -2)[-2]) if probs.size >= 2 else 0.0
    margin = conf - second
    top = label_map[str(idx)]
    accepted = conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD
    reason = ""
    if not accepted:
        if conf < CONFIDENCE_THRESHOLD:
            reason = "low confidence"
        else:
            reason = "ambiguous"
    return {"top": top, "conf": conf, "second": second, "margin": margin,
            "accepted": accepted, "reason": reason}


def classify_segment(model, label_map, segment, prev_label=None):
    """Resample a raw variable-length segment to SEQ_LEN, normalize, predict.
    Returns (label or None, confidence). Applies confidence + margin gates.
    `prev_label` (the last committed sign) feeds the context-prior hook."""
    d = classify_segment_debug(model, label_map, segment, prev_label)
    return (d["top"] if d["accepted"] else None), d["conf"]


def main():
    model = load_model(MODEL_FILE)
    with open(LABELS_FILE) as f:
        label_map = json.load(f)

    speaker = Speaker(rate=TTS_RATE, enabled=TTS_ENABLED)
    if TTS_ENABLED and not speaker.available:
        print("[TTS] pyttsx3 not installed — running without speech. "
              "Install with: pip install pyttsx3")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    WINDOW = "Motion Sign Translator"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    print("Controls: 'c' = clear sentence | 't' = toggle voice | 'q' = quit")

    preroll = deque(maxlen=PREROLL)   # raw frames kept while idle
    segment = []                       # raw frames of the sign being recorded
    collecting = False
    still_count = 0
    start_count = 0                    # consecutive frames of motion >= MOTION_START
    prev_raw = None

    sentence = []          # list of (word, confidence) for accepted signs
    last_sign = ""         # most recent recognized sign text (status line)
    last_conf = 0.0        # its confidence (colors the status line)
    last_activity = time.time()   # for auto-clear after an idle pause
    last_added_at = 0.0    # when the last sign was appended (for repeat handling)
    # why-overlay: diagnostics of the most recent classified segment
    last_reason = ""       # "" = translated; else low confidence / ambiguous / not a sign
    last_reason_top = ""   # the model's best guess (even when rejected)
    last_reason_conf = 0.0
    last_reason_at = 0.0   # when it was set (the HUD line fades after a few seconds)

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

        raw = hf.extract_raw_hands(results)
        hand_present = any(v != 0.0 for v in raw)
        motion = hf.frame_motion(prev_raw, raw) if prev_raw is not None else 0.0

        finished = None  # set to a segment (list of raw frames) when a sign ends

        if not collecting:
            preroll.append(raw)
            # count consecutive above-threshold frames; a short run (not a single
            # spike) starts the segment. Any settled/hand-absent frame resets it.
            if hand_present and motion >= MOTION_START:
                start_count += 1
            else:
                start_count = 0
            if start_count >= START_FRAMES:
                collecting = True
                segment = list(preroll)   # include a little pre-motion context
                still_count = 0
                start_count = 0
        else:
            last_activity = time.time()   # actively signing counts as activity
            segment.append(raw)
            if motion < MOTION_STOP or not hand_present:
                still_count += 1
            else:
                still_count = 0
            if still_count >= STOP_FRAMES or len(segment) >= MAX_SEG:
                finished = segment
                collecting = False
                segment = []
                preroll.clear()

        if finished is not None and len(finished) >= MIN_SEG:
            d = classify_segment_debug(model, label_map, finished)
            top_l = d["top"].strip().lower()
            is_ignore = top_l in IGNORE_LABELS
            label = d["top"] if (d["accepted"] and not is_ignore) else None
            conf = d["conf"]
            # remember why the last segment did / didn't translate (on-screen HUD)
            if d["accepted"] and is_ignore:
                last_reason = "not a sign"
            elif d["accepted"]:
                last_reason = ""
            else:
                last_reason = d["reason"]
            last_reason_top = d["top"]
            last_reason_conf = d["conf"]
            last_reason_at = time.time()
            if label is not None:
                sign = label.replace("_", " ")
                now = time.time()
                last_sign, last_conf = sign, conf
                last_activity = now
                # Add the sign, unless it's an immediate repeat of the previous
                # one within REPEAT_GAP_SECONDS (that's the same sign detected
                # twice, not a deliberate re-sign). A repeat after a real pause
                # IS added, so signing "Hi Hi" on purpose shows both.
                is_repeat = bool(sentence) and sentence[-1][0] == sign
                if not is_repeat or (now - last_added_at) >= REPEAT_GAP_SECONDS:
                    sentence.append((sign, conf))
                    del sentence[:-MAX_SENTENCE]
                    last_added_at = now
                    speaker.say(sign)          # speak the newly committed sign
                else:
                    # refresh the confidence shown for the existing last token
                    sentence[-1] = (sign, conf)

        # auto-clear the sentence after an idle pause (only when not signing)
        if sentence and not collecting and (time.time() - last_activity) > AUTO_CLEAR_SECONDS:
            sentence.clear()
            last_sign = ""      # keep the corner "last:" consistent with the caption
            last_conf = 0.0

        # status line (top-left)
        if collecting:
            status, color = f"SIGNING...  {len(segment)} frames", COLOR_CURRENT
        else:
            status, color = "READY - start signing", COLOR_HIGH
        outlined(frame, status, (10, 30), 0.7, color, 2)
        # live motion readout: shows whether you're crossing MOTION_START. If this
        # stays below the threshold while you sign, the segment never starts — sign
        # a touch faster/larger or lower MOTION_START. Green once over threshold.
        mot_col = COLOR_HIGH if motion >= MOTION_START else (150, 150, 150)
        mot_txt = f"motion {motion:.3f} / start {MOTION_START:.3f}"
        (mtw, _), _ = cv2.getTextSize(mot_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        outlined(frame, mot_txt, (frame.shape[1] - mtw - 12, 56), 0.5, mot_col, 1)
        if last_sign:
            outlined(frame, f"last: {last_sign} ({last_conf * 100:.0f}%)",
                     (10, 58), 0.6, conf_color(last_conf), 1)
        if speaker.available:
            tts_txt = "voice: ON" if speaker.enabled else "voice: OFF"
            tts_col = COLOR_HIGH if speaker.enabled else (150, 150, 150)
            outlined(frame, tts_txt, (10, 82), 0.5, tts_col, 1)
            outlined(frame, "c=clear  t=voice  q=quit", (10, 104), 0.5, (180, 180, 180), 1)
        else:
            outlined(frame, "c=clear sentence   q=quit", (10, 82), 0.5, (180, 180, 180), 1)

        # committed signs colored by confidence + a cyan marker while recording
        tokens = [(word, conf_color(c)) for word, c in sentence]
        if collecting:
            tokens.append(("...", COLOR_CURRENT))
        draw_subtitle(frame, tokens)

        # why-overlay (top-right): show why the last segment did/didn't translate,
        # for a few seconds after each classification. Rejected segments are the
        # useful case — you can see if it was low confidence vs. an ambiguous
        # near-miss vs. read as the "not a sign" class.
        if last_reason and (time.time() - last_reason_at) < 4.0:
            if last_reason == "not a sign":
                why = "ignored: not a sign"
            else:
                why = (f"rejected ({last_reason}): {last_reason_top} "
                       f"{last_reason_conf * 100:.0f}%")
            (tw, _), _ = cv2.getTextSize(why, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            outlined(frame, why, (frame.shape[1] - tw - 12, 30), 0.5,
                     (0, 160, 255), 1)

        cv2.imshow(WINDOW, frame)
        prev_raw = raw

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
            last_sign = ""
            last_conf = 0.0
        elif key == ord("t"):
            state = speaker.toggle()
            print(f"[TTS] voice {'on' if state else 'off'}")

    speaker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
