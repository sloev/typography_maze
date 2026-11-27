"""The walk: one stroke per sentence.

2D: the stroke's length is the sentence's word count; before drawing it the path
turns 90 degrees left for a positive sentence, right for a negative one, and goes
straight for a neutral one. Coordinates are integers (word units), so crossings
and overlaps are exact.

3D ("mineshaft"): the same horizontal turns, and in addition every left turn
tilts the path up by DELTA degrees and every right turn tilts it down, so the
elevation angle is DELTA x (lefts - rights so far), clamped to +/- PITCH_MAX.
"""
from dataclasses import dataclass
import math
import numpy as np

DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])


@dataclass
class Walk:
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    d: np.ndarray       # heading index 0..3 (0 = initial direction)
    n: np.ndarray       # stroke length (words)
    turn: np.ndarray    # -1 right, 0 straight, +1 left
    score: np.ndarray   # sentence sentiment score

    @property
    def N(self):
        return len(self.n)

    def vertices(self):
        return np.concatenate([[[self.x0[0], self.y0[0]]], np.stack([self.x1, self.y1], 1)])


def walk2d(lengths, turns, scores=None):
    lengths = np.asarray(lengths, int)
    turns = np.asarray(turns, int)
    d = np.cumsum(turns) % 4
    step = DIRS[d] * lengths[:, None]
    ends = np.cumsum(step, axis=0)
    starts = np.vstack([[0, 0], ends[:-1]])
    sc = np.zeros(len(lengths)) if scores is None else np.asarray(scores, float)
    return Walk(starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1], d, lengths, turns, sc)


def walk3d(lengths, turns, delta_deg=0.75, pitch_max_deg=80.0):
    """Returns (N+1, 3) vertices and the per-stroke pitch in degrees."""
    lengths = np.asarray(lengths, float)
    turns = np.asarray(turns, int)
    heading = (np.cumsum(turns) % 4) * (math.pi / 2)
    pitch = np.clip(np.cumsum(turns) * delta_deg, -pitch_max_deg, pitch_max_deg)
    p = np.radians(pitch)
    step = np.stack([np.cos(heading) * np.cos(p), np.sin(heading) * np.cos(p), np.sin(p)], 1) * lengths[:, None]
    v = np.vstack([[0, 0, 0], np.cumsum(step, axis=0)])
    return v, pitch
