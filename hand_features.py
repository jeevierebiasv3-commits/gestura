"""
hand_features.py
----------------
Single source of truth for turning MediaPipe hand-detection results into the
feature vector the model consumes. Imported by collect_data.py, train_model.py,
app.py, and live_translate.py so the *exact same* logic runs at data-collection,
training, and inference time (no train/serve skew).

KEY IDEA — why the old app was inaccurate
------------------------------------------
The old code fed the model MediaPipe's raw image-space (x, y) coordinates. Those
encode WHERE the hand sits in the frame and HOW BIG it looks (distance to the
camera), not the hand's SHAPE. So moving or changing distance broke predictions.

`normalize()` fixes this by, for each hand independently:
  1. Translating so the wrist (landmark 0) is the origin  -> position invariance
  2. Scaling by the wrist -> middle-finger-MCP distance    -> distance invariance
Rotation is intentionally left intact because orientation carries meaning in
sign language (e.g. thumb up vs thumb down).

LAYOUT
------
Raw feature vector = 126 numbers:
  Left  hand: 21 landmarks x (x, y, z) = 63 numbers  (indices  0..62)
  Right hand: 21 landmarks x (x, y, z) = 63 numbers  (indices 63..125)
A missing hand is all zeros for its 63-slot.
"""

import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # allows importing constants without mediapipe installed
    _MP_AVAILABLE = False

NUM_LANDMARKS = 21          # MediaPipe hand landmarks per hand
COORDS = 3                  # x, y, z
PER_HAND = NUM_LANDMARKS * COORDS   # 63
RAW_DIM = PER_HAND * 2              # 126  (left + right)
FEATURE_DIM = RAW_DIM              # what the model's Input layer expects

WRIST = 0                  # landmark index of the wrist
MIDDLE_MCP = 9             # landmark index of the middle-finger knuckle (scale ref)

SEQ_LEN = 45               # frames per motion clip (Stage 2 sequence model).
                           # Any real-duration clip is resampled to this length,
                           # so a longer value keeps more temporal detail for
                           # long signs at the cost of a bit more compute.

# --- Stage-2 MOTION channels (see motion_channels / seq_features below) ---
# Per-frame shape features (126) tell the model WHAT the hand looks like but,
# because normalize() pins each hand's wrist to the origin every frame, they say
# nothing about WHERE the hand traveled. These extra channels restore that: for
# each hand, the wrist's displacement-from-clip-start and its per-frame velocity
# (both anchored + scaled so they stay position/distance invariant like the shape
# features). This is what lets the GRU tell a still sign (e.g. "I") apart from a
# moving one (e.g. "Sorry" — circular motion on the chest).
MOTION_PER_HAND = 6                       # displacement(x,y,z) + velocity(x,y,z)
MOTION_DIM = MOTION_PER_HAND * 2          # 12  (left + right)
SEQ_FEATURE_DIM = RAW_DIM + MOTION_DIM    # 138 — the Stage-2 model's input width
# column layout inside seq_features(): [shape 0..125][Left motion 126..131][Right motion 132..137]

_EPS = 1e-6


def make_hands(static_image_mode=False):
    """Create a MediaPipe Hands tracker with the project's standard settings.

    Lower detection threshold so hands aren't dropped easily; higher tracking
    threshold so a locked-on hand sticks around.
    """
    if not _MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed")
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7,
        model_complexity=1,
    )


def csv_header():
    """Column names for data.csv: 126 landmark columns + 'label'."""
    cols = []
    for hand in ("L", "R"):
        for i in range(NUM_LANDMARKS):
            cols += [f"{hand}{i}_x", f"{hand}{i}_y", f"{hand}{i}_z"]
    cols.append("label")
    return cols


def extract_raw_hands(results):
    """MediaPipe results -> flat list of 126 raw floats (Left slot, Right slot).

    Includes the z (depth) coordinate. Missing hand -> zeros for its slot.
    """
    left = [0.0] * PER_HAND
    right = [0.0] * PER_HAND

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left = coords
            else:
                right = coords

    return left + right  # 126 numbers


def _normalize_hand(block):
    """Normalize one hand's 63-number block. All-zeros (absent) stays zeros."""
    pts = np.asarray(block, dtype="float32").reshape(NUM_LANDMARKS, COORDS)
    if not np.any(pts):
        return pts.reshape(-1)

    # 1) translate: wrist -> origin
    pts = pts - pts[WRIST]

    # 2) scale: divide by wrist -> middle-MCP distance (in xy, the stable ref)
    scale = np.linalg.norm(pts[MIDDLE_MCP][:2])
    if scale < _EPS:
        scale = np.linalg.norm(pts[MIDDLE_MCP]) or 1.0
    pts = pts / scale

    return pts.reshape(-1)


def normalize(raw_vec):
    """Turn a raw 126-vector into a position/scale-invariant 126-vector."""
    raw = np.asarray(raw_vec, dtype="float32").reshape(-1)
    left = _normalize_hand(raw[:PER_HAND])
    right = _normalize_hand(raw[PER_HAND:])
    return np.concatenate([left, right]).astype("float32")


def mirror(vec):
    """Left-right mirror: swap the two hand slots and negate every x coordinate.

    Works on raw OR normalized vectors (the layout is preserved). Used for
    augmentation so the model tolerates MediaPipe's handedness flipping under
    the capture mirror.
    """
    v = np.asarray(vec, dtype="float32").reshape(2, NUM_LANDMARKS, COORDS).copy()
    v[..., 0] *= -1.0          # negate x
    v = v[::-1]                # swap left <-> right slots
    return v.reshape(-1)


def _augment_one(norm_vec, rng):
    """Apply small random rotation / scale / jitter / translation to a
    normalized vector. Empty (all-zero) hand slots are left untouched."""
    v = np.asarray(norm_vec, dtype="float32").reshape(2, NUM_LANDMARKS, COORDS).copy()

    angle = rng.uniform(-0.26, 0.26)          # ~±15 degrees, in-plane tilt
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype="float32")
    scale = rng.uniform(0.9, 1.1)
    trans = rng.uniform(-0.05, 0.05, size=2).astype("float32")

    for h in range(2):
        hand = v[h]
        if not np.any(hand):
            continue                          # keep absent hand as zeros
        hand[:, :2] = hand[:, :2] @ rot.T     # rotate xy
        hand *= scale                          # scale xyz
        hand[:, :2] += trans                   # translate xy
        hand += rng.normal(0.0, 0.01, size=hand.shape).astype("float32")  # jitter
        v[h] = hand

    return v.reshape(-1)


def augment(raw_vec, n_aug=6, include_mirror=True, seed=None):
    """Expand one raw sample into many normalized training vectors.

    Returns the normalized original plus `n_aug` randomized variants, and (if
    include_mirror) the same set again mirrored. Feed the raw data.csv rows in;
    get model-ready normalized features out.
    """
    rng = np.random.default_rng(seed)
    base = normalize(raw_vec)
    out = [base]
    out += [_augment_one(base, rng) for _ in range(n_aug)]

    if include_mirror:
        m = mirror(base)
        out.append(m)
        out += [_augment_one(m, rng) for _ in range(n_aug)]

    return np.asarray(out, dtype="float32")


# --------------------------------------------------------------------------
# Stage 2 helpers: motion / sequence signs
# --------------------------------------------------------------------------

def resample_sequence(frames, length=SEQ_LEN):
    """Resample a variable-length clip of raw 126-vectors to exactly `length`
    frames via linear interpolation along time. Input/returns (T, 126)."""
    arr = np.asarray(frames, dtype="float32")
    if arr.ndim != 2 or arr.shape[1] != RAW_DIM:
        raise ValueError(f"expected (T, {RAW_DIM}), got {arr.shape}")
    t = arr.shape[0]
    if t == length:
        return arr
    if t == 1:
        return np.repeat(arr, length, axis=0)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, RAW_DIM), dtype="float32")
    for c in range(RAW_DIM):
        out[:, c] = np.interp(dst, src, arr[:, c])
    return out


def normalize_sequence(frames):
    """Normalize every frame of a clip (T, 126) -> (T, 126)."""
    arr = np.asarray(frames, dtype="float32")
    return np.stack([normalize(f) for f in arr]).astype("float32")


def motion_channels(frames):
    """Per-frame wrist MOTION features for a fixed-length clip (T, 126) -> (T, 12).

    Restores the trajectory information that per-frame normalize() removes (it
    pins each hand's wrist to the origin every frame, erasing where the hand
    traveled). For each hand (Left, then Right) and each frame, 6 numbers:
        displacement (x,y,z): wrist position minus the clip's start anchor
        velocity     (x,y,z): wrist position minus the previous frame's wrist
    Both are divided by the hand's size (median wrist->middle-MCP xy distance over
    the clip) so they stay distance-invariant, matching the shape features. The
    anchor makes displacement position-invariant.

    Robustness rules:
      - anchor = wrist at the FIRST frame where that hand is present (non-zero).
      - a frame where the hand is absent -> that hand's 6 channels stay 0.
      - velocity is taken only when BOTH this frame and the previous one have the
        hand present, so a hand (re)appearing can't create a huge false spike.
      - a hand present in < 2 frames -> all 6 of its channels stay 0.
    Column layout matches seq_features(): [Left 6][Right 6]."""
    arr = np.asarray(frames, dtype="float32")
    if arr.ndim != 2 or arr.shape[1] != RAW_DIM:
        raise ValueError(f"expected (T, {RAW_DIM}), got {arr.shape}")
    T = arr.shape[0]
    out = np.zeros((T, MOTION_DIM), dtype="float32")

    for h in range(2):
        base = h * PER_HAND
        block = arr[:, base:base + PER_HAND].reshape(T, NUM_LANDMARKS, COORDS)
        present = np.any(block != 0.0, axis=(1, 2))      # (T,) hand visible?
        if int(present.sum()) < 2:
            continue                                      # can't define motion
        wrist = block[:, WRIST, :]                        # (T, 3)
        mcp = block[:, MIDDLE_MCP, :]                     # (T, 3)

        # hand size: median wrist->middle-MCP xy distance over present frames
        dists = np.linalg.norm((mcp - wrist)[:, :2], axis=1)[present]
        scale = float(np.median(dists))
        if scale < _EPS:
            scale = 1.0

        anchor = wrist[present][0]                         # first present wrist
        col = h * MOTION_PER_HAND
        for t in range(T):
            if not present[t]:
                continue
            out[t, col:col + 3] = (wrist[t] - anchor) / scale        # displacement
            if t > 0 and present[t - 1]:                             # velocity, only
                out[t, col + 3:col + 6] = (wrist[t] - wrist[t - 1]) / scale
    return out


def seq_features(frames):
    """Full Stage-2 per-frame features for a fixed clip (T, 126) -> (T, 138).

    THE single source of truth for the model's input column order:
        [ normalized shape 126 ][ motion 12 ]
    Used at inference (live_sequence.classify_segment_debug) and to build the
    test set (train_sequence_model); augment_sequence() emits the SAME layout for
    training, so train and serve can never disagree on what each column means."""
    shape = normalize_sequence(frames)                    # (T, 126)
    motion = motion_channels(frames)                      # (T, 12)
    return np.concatenate([shape, motion], axis=1).astype("float32")


def frame_motion(prev_raw, cur_raw):
    """Mean per-coordinate movement between two raw 126-vectors.

    Only coordinates where BOTH frames have a hand present (non-zero) are
    counted, so an appearing/disappearing hand doesn't dominate. Values are in
    MediaPipe's normalized image space (x, y in [0, 1]); a still hand gives
    ~0.001-0.005 (jitter), active signing gives ~0.02+. Used to detect when the
    user is actively signing vs. holding still (segmentation for long/continuous
    signs).

    Slot-flip robustness: MediaPipe's Left/Right label for a SINGLE hand in the
    mirrored view is unstable and flips frame-to-frame. On a flip the hand jumps
    between the Left and Right slots, so no coordinate is non-zero in both frames
    and the naive both-present mask reads 0.0 — a moving hand looks still and the
    segmenter never starts. When that happens we fall back to matching the one
    present hand across slots so real motion is still measured.
    """
    a = np.asarray(prev_raw, dtype="float32")
    b = np.asarray(cur_raw, dtype="float32")
    mask = (a != 0.0) & (b != 0.0)
    if np.any(mask):
        return float(np.mean(np.abs(a[mask] - b[mask])))

    # No slot lines up: either a hand genuinely appeared/disappeared, or a single
    # hand flipped Left<->Right between the frames. Match the present block in
    # each frame regardless of slot; if either frame has no hand at all, it's a
    # real appear/disappear and stays 0.0.
    a_block = a[:PER_HAND] if np.any(a[:PER_HAND]) else a[PER_HAND:]
    b_block = b[:PER_HAND] if np.any(b[:PER_HAND]) else b[PER_HAND:]
    if not (np.any(a_block) and np.any(b_block)):
        return 0.0
    return float(np.mean(np.abs(a_block - b_block)))



def augment_sequence(frames, n_aug=4, include_mirror=True, seed=None):
    """Expand one fixed-length RAW clip (T, 126) into several model-ready
    augmented clips (n, T, 138) — shape + motion, matching seq_features()'s
    column order.

    Shape (126) and motion (12) channels are transformed TOGETHER so they stay
    geometrically consistent within each variant:
      - one shared in-plane rotation rotates the xy of the shape landmarks AND of
        the wrist displacement/velocity vectors.
      - one shared scale (xyz) is applied to both (matching robustness jitter).
      - translation + positional jitter hit SHAPE ONLY: motion is a difference of
        positions, so it's translation-invariant by construction and must not be
        shifted or jittered.
      - mirroring negates x and swaps the Left/Right hand slots for BOTH shape and
        motion. The L/R swap MUST happen for motion too, otherwise shape-left gets
        paired with motion-right and every mirrored clip is mislabeled.
    """
    rng = np.random.default_rng(seed)
    raw = np.asarray(frames, dtype="float32")
    shape_base = normalize_sequence(raw)          # (T, 126)
    motion_base = motion_channels(raw)            # (T, 12)
    T = shape_base.shape[0]

    def mirror_shape(clip):
        return np.stack([mirror(f) for f in clip])

    def mirror_motion(clip):
        m = clip.reshape(T, 2, 2, COORDS).copy()  # (T, hand, {disp,vel}, xyz)
        m[..., 0] *= -1.0                          # negate x of disp & vel
        m = m[:, ::-1]                             # swap Left <-> Right hand blocks
        return m.reshape(T, MOTION_DIM)

    def transform(do_mirror):
        s = mirror_shape(shape_base) if do_mirror else shape_base.copy()
        m = mirror_motion(motion_base) if do_mirror else motion_base.copy()
        angle = rng.uniform(-0.26, 0.26)
        cs, sn = np.cos(angle), np.sin(angle)
        rot = np.array([[cs, -sn], [sn, cs]], dtype="float32")
        scale = rng.uniform(0.9, 1.1)
        trans = rng.uniform(-0.05, 0.05, size=2).astype("float32")

        # shape: rotate xy, scale xyz, translate xy, jitter
        s = s.reshape(T, 2, NUM_LANDMARKS, COORDS)
        for fi in range(T):
            for h in range(2):
                hand = s[fi, h]
                if not np.any(hand):
                    continue
                hand[:, :2] = hand[:, :2] @ rot.T
                hand *= scale
                hand[:, :2] += trans
                s[fi, h] = hand
        s += rng.normal(0.0, 0.01, size=s.shape).astype("float32")
        s = s.reshape(T, RAW_DIM)

        # motion: rotate xy, scale xyz — NO translate, NO jitter
        mm = m.reshape(T, 2, 2, COORDS)
        mm[..., :2] = mm[..., :2] @ rot.T
        mm *= scale
        m = mm.reshape(T, MOTION_DIM)

        return np.concatenate([s, m], axis=1).astype("float32")

    clips = [np.concatenate([shape_base, motion_base], axis=1).astype("float32")]
    for _ in range(n_aug):
        clips.append(transform(do_mirror=False))
    if include_mirror:
        clips.append(np.concatenate(
            [mirror_shape(shape_base), mirror_motion(motion_base)], axis=1
        ).astype("float32"))
        for _ in range(n_aug):
            clips.append(transform(do_mirror=True))

    return np.asarray(clips, dtype="float32")
