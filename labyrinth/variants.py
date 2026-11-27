"""Variations of the turning rule.

    quarter  the book's rule: each sentence turns 90 degrees left (positive),
             right (negative) or not at all (neutral).
    graded   each sentence turns by its strength of feeling: v x 90 degrees, where
             v = S / sqrt(S^2 + 15) is the sentence's AFINN sum S squashed into
             (-1, 1) with VADER's normalisation. A sentence at v = 0.1 barely
             bends; one at v = 0.9 turns nearly a right angle.
    rolling  every word is a step of one unit, and the heading bends at every
             word by the running mean of the last WINDOW words' valences
             (AFINN / 5), times GAIN degrees. The path curves as the
             passage reads, instead of kinking at sentence ends.

Sign: positive turns left (counter-clockwise), as everywhere in the book.
SIGN = -1 mirrors every drawing (positive turns right).

All rules also give the mineshaft: the path tilts up by DELTA degrees for every
90 degrees turned left, and down for every 90 degrees turned right.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import sentiment

SIGN = +1
DELTA = 0.75          # tilt in degrees per 90 degrees of turning
PITCH_MAX = 80.0
WINDOW = 12           # rolling window, words
GAIN = 40.0           # rolling: degrees per word at a running valence of 1


def valence(S):
    return S / math.sqrt(S * S + 15.0)


@dataclass
class Path:
    """Strokes with arbitrary headings. Lengths in the units given."""
    rule: str
    heading: np.ndarray   # degrees, per stroke
    length: np.ndarray    # per stroke
    text: list            # per stroke (sentence or word)
    group: np.ndarray     # sentence index per stroke
    turn_deg: np.ndarray  # turn made before each stroke

    def vertices2d(self):
        h = np.radians(self.heading)
        step = np.stack([np.cos(h), np.sin(h)], 1) * self.length[:, None]
        return np.vstack([[0, 0], np.cumsum(step, 0)])

    def vertices3d(self, delta=DELTA, pitch_max=PITCH_MAX):
        pitch = np.clip(np.cumsum(self.turn_deg) / 90.0 * delta, -pitch_max, pitch_max)
        h, p = np.radians(self.heading), np.radians(pitch)
        step = np.stack([np.cos(h) * np.cos(p), np.sin(h) * np.cos(p), np.sin(p)], 1) * self.length[:, None]
        return np.vstack([[0, 0, 0], np.cumsum(step, 0)]), pitch

    def with_lengths(self, lengths):
        return Path(self.rule, self.heading, np.asarray(lengths, float), self.text, self.group, self.turn_deg)


def build(book, rule):
    S = book.sentences
    names = sentiment.book_names([s.text for s in S])
    if rule in ("quarter", "graded"):
        sums = np.array([sentiment.score_text(s.text, names) for s in S], float)
        if rule == "quarter":
            turn = np.array([sentiment.turn(x) for x in sums], float) * 90.0
        else:
            turn = np.array([valence(x) for x in sums]) * 90.0
        turn *= SIGN
        heading = np.cumsum(turn)
        return Path(rule, heading, np.array([len(s.words) for s in S], float), [s.text for s in S],
                    np.arange(len(S)), turn)
    if rule == "rolling":
        words, groups, vals = [], [], []
        for s in S:
            toks, v = sentiment.token_valences(s.text, names)
            words += toks; groups += [s.i] * len(toks); vals += [x / 5.0 for x in v]
        vals = np.array(vals)
        c = np.concatenate([[0], np.cumsum(vals)])
        k = np.arange(1, len(vals) + 1)
        lo = np.maximum(0, k - WINDOW)
        rolling = (c[k] - c[lo]) / (k - lo)
        turn = SIGN * GAIN * rolling
        heading = np.cumsum(turn)
        return Path(rule, heading, np.ones(len(words)), words, np.array(groups), turn)
    raise ValueError(rule)


RULES = ("quarter", "graded", "rolling")
