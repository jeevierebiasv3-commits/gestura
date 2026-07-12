"""
suggest.py  —  next-sign autocomplete + a shared transition model
-----------------------------------------------------------------
Given the signs committed so far, propose the few signs the user is most likely
to reach for next (like the word suggestions above a phone keyboard), and expose
the same knowledge as a probability prior the recognizer can use to re-rank its
softmax toward grammatically-likely continuations.

This is a small **bigram model** ("how often does sign B follow sign A"):

  - It ships with a hand-authored SEED corpus of realistic sign orderings so it
    is useful from the very first run (cold-start), scoped to the current vocab.
  - It LEARNS from real usage: every committed (prev -> current) pair is folded
    into a persisted counts file (COUNTS_FILE) via record_transition(), so the
    suggestions and the prior get better the more the app is used.

Two consumers share this one model (single source of truth — no drift):
  - next_signs()       -> the autocomplete chips (app_sequence.py)
  - transition_prior() -> apply_context_prior() in live_sequence.py

Everything is best-effort and degrades gracefully: a missing/corrupt counts file,
an unknown previous sign, or a disk-write failure never raises — suggestions just
fall back to popularity and the prior falls back to "no opinion" (None).
"""

import os
import json
import threading

# Where to persist learned counts. In dev this is just the file in the cwd
# (unchanged behaviour). When packaged as a desktop app the launcher sets
# GESTURA_DATA_DIR to a WRITABLE folder next to the .exe, because the bundled
# resources are read-only — otherwise online learning would silently fail.
_DATA_DIR = os.environ.get("GESTURA_DATA_DIR", "")
COUNTS_FILE = (os.path.join(_DATA_DIR, "suggest_counts.json")
               if _DATA_DIR else "suggest_counts.json")

# --- SEED corpus: realistic next-sign orderings within the current vocabulary.
# Authored pairs (previous sign -> likely next sign). These give sensible
# suggestions before any real usage is logged; learned counts add on top.
SEED_PAIRS = [
    ("Hi", "Good Morning"), ("Hi", "Good Afternoon"), ("Hi", "Hello"),
    ("Hello", "Good Morning"), ("Hello", "Good Afternoon"),
    ("Good Morning", "Thank you"), ("Good Afternoon", "Thank you"),
    ("Excuse me", "Sorry"), ("Excuse me", "Thank you"),
    ("Sorry", "Thank you"), ("Sorry", "Excuse me"),
    ("I", "Sorry"), ("I", "Thank you"), ("I", "Yes"),
    ("I love you", "Take care"), ("I love you", "See you later"),
    ("Thank you", "Take care"), ("Thank you", "See you later"),
    ("Thank you", "Yes"),
    ("Well done", "Congratulations"), ("Well done", "Thank you"),
    ("Congratulations", "Well done"), ("Congratulations", "Thank you"),
    ("Take care", "See you later"), ("See you later", "Take care"),
    ("Yes", "Thank you"), ("Yes", "Well done"),
]

# how much a seed pair "weighs" relative to one real observed transition
SEED_WEIGHT = 2
# Laplace smoothing added to every candidate when forming a probability prior,
# so a plausible-but-unseen continuation is never assigned exactly zero.
PRIOR_SMOOTHING = 0.4

_lock = threading.Lock()
# bigrams[prev_lower][curr_canonical] = count ; unigram[curr_canonical] = count
_bigrams = {}
_unigram = {}
_loaded = False


def _bump(bigrams, unigram, prev, curr, n):
    p = prev.strip().lower()
    inner = bigrams.setdefault(p, {})
    inner[curr] = inner.get(curr, 0) + n
    unigram[curr] = unigram.get(curr, 0) + n


def _load():
    """Build the model from SEED + any persisted usage counts (once)."""
    global _loaded
    if _loaded:
        return
    _bigrams.clear()
    _unigram.clear()
    for prev, curr in SEED_PAIRS:
        _bump(_bigrams, _unigram, prev, curr, SEED_WEIGHT)
    # merge learned usage on top (best-effort — ignore a missing/corrupt file)
    try:
        if os.path.isfile(COUNTS_FILE):
            with open(COUNTS_FILE) as f:
                data = json.load(f)
            for prev, inner in (data.get("bigrams") or {}).items():
                for curr, n in inner.items():
                    _bump(_bigrams, _unigram, prev, curr, int(n))
    except Exception:
        pass
    _loaded = True


def _persist_learned(prev, curr):
    """Append one observed transition to COUNTS_FILE (best-effort)."""
    try:
        data = {"bigrams": {}}
        if os.path.isfile(COUNTS_FILE):
            with open(COUNTS_FILE) as f:
                data = json.load(f)
        bg = data.setdefault("bigrams", {})
        inner = bg.setdefault(prev.strip().lower(), {})
        inner[curr] = int(inner.get(curr, 0)) + 1
        tmp = COUNTS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, COUNTS_FILE)   # atomic swap so a crash can't corrupt it
    except Exception:
        pass                            # never let logging break the app


def record_transition(prev, curr):
    """Learn one (prev -> curr) transition from real usage.

    Called when a new sign is committed after a previous one. Updates the
    in-memory model immediately and persists it. Best-effort; never raises."""
    if not prev or not curr:
        return
    with _lock:
        _load()
        _bump(_bigrams, _unigram, prev, curr, 1)
        _persist_learned(prev, curr)


def _popular(k=None):
    """Signs ordered by overall frequency (unigram counts), most first."""
    _load()
    ranked = sorted(_unigram, key=lambda c: _unigram[c], reverse=True)
    return ranked[:k] if k else ranked


def next_signs(words, vocab, k=3):
    """Return up to k suggested next signs (canonical-cased, drawn from `vocab`).

    words : the sentence so far (most recent sign last)
    vocab : the full list of real sign labels (e.g. seq_labels.json values)
    Ranks by learned/seeded bigram counts for the previous sign, backing off to
    overall popularity; never repeats the current last sign."""
    with _lock:
        _load()
        canon = {v.strip().lower(): v for v in vocab}
        prev = words[-1].strip().lower() if words else ""

        ranked = []
        inner = _bigrams.get(prev, {})
        for cand in sorted(inner, key=lambda c: inner[c], reverse=True):
            ranked.append(cand)
        # back off to popularity to top up a short/empty transition list
        for cand in _popular():
            if cand not in ranked:
                ranked.append(cand)

        out = []
        for cand in ranked:
            c = canon.get(cand.strip().lower())
            if c and c.strip().lower() != prev and c not in out:
                out.append(c)
            if len(out) >= k:
                break
        return out[:k]


def transition_prior(prev_label, labels):
    """A probability prior over `labels` given the previous sign, or None.

    Returns a list of floats (summing to 1) aligned to `labels`, biasing toward
    signs that commonly follow `prev_label`. Returns None when there is no
    information for prev_label, so callers can treat the prior as "no opinion"
    and leave their prediction untouched (a true no-op, not a flattening)."""
    if not prev_label:
        return None
    with _lock:
        _load()
        inner = _bigrams.get(prev_label.strip().lower())
        if not inner:
            return None
        # canonical-label -> count lookup (case-insensitive)
        by_lower = {}
        for cand, n in inner.items():
            by_lower[cand.strip().lower()] = by_lower.get(cand.strip().lower(), 0) + n
        raw = [by_lower.get(l.strip().lower(), 0) + PRIOR_SMOOTHING for l in labels]
        total = sum(raw)
        if total <= 0:
            return None
        return [r / total for r in raw]
