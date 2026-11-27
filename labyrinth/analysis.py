"""The labyrinthographic toolkit.

Every measure is computed from the walk alone. Measures that depend on the
ORDER of the sentences are tested against a shuffle baseline: the same
sentences (same lengths, same sentiment) redrawn in random order. The baseline
keeps everything about the book except its sequence, so a z-score far from 0
says the drawing's shape comes from how the book is ordered, not from what it
is made of.
"""
import math
from collections import Counter

import numpy as np

from .walk import walk2d

# ---------------------------------------------------------------- crossings


def crossings(w):
    """All proper crossings between a horizontal and a vertical stroke.

    Returns an array of rows (x, y, i, j) with i < j, the two sentence indices.
    Strokes that are neighbours in the text (|i - j| = 1) share an endpoint and
    are not counted. Crossings are interior: touching at an endpoint is not a
    crossing.
    """
    horiz = w.d % 2 == 0
    hi = np.nonzero(horiz)[0]
    vi = np.nonzero(~horiz)[0]
    hy = w.y0[hi]
    hx0 = np.minimum(w.x0[hi], w.x1[hi]); hx1 = np.maximum(w.x0[hi], w.x1[hi])
    vx = w.x0[vi]
    vy0 = np.minimum(w.y0[vi], w.y1[vi]); vy1 = np.maximum(w.y0[vi], w.y1[vi])
    order = np.argsort(vx, kind="stable")
    vx, vy0, vy1, vi = vx[order], vy0[order], vy1[order], vi[order]
    out = []
    for k in range(len(hi)):
        a = np.searchsorted(vx, hx0[k], "right")
        b = np.searchsorted(vx, hx1[k], "left")
        if a >= b:
            continue
        sel = slice(a, b)
        m = (vy0[sel] < hy[k]) & (vy1[sel] > hy[k])
        if not m.any():
            continue
        js = vi[sel][m]
        xs = vx[sel][m]
        i = hi[k]
        keep = np.abs(js - i) > 1
        for x, j in zip(xs[keep], js[keep]):
            out.append((x, hy[k], min(i, j), max(i, j)))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def overlaps(w):
    """Collinear overlaps (palimpsests): a later sentence lying exactly on an earlier one.

    Returns rows (i, j, length, x0, y0, x1, y1) for every overlapping pair.
    """
    res = []
    for horiz in (True, False):
        idx = np.nonzero((w.d % 2 == 0) == horiz)[0]
        if horiz:
            line = w.y0[idx]; a = np.minimum(w.x0[idx], w.x1[idx]); b = np.maximum(w.x0[idx], w.x1[idx])
        else:
            line = w.x0[idx]; a = np.minimum(w.y0[idx], w.y1[idx]); b = np.maximum(w.y0[idx], w.y1[idx])
        order = np.lexsort((a, line))
        line, a, b, idx = line[order], a[order], b[order], idx[order]
        n = len(idx)
        k = 0
        while k < n:
            m = k
            while m < n and line[m] == line[k]:
                m += 1
            for p in range(k, m):
                for q in range(p + 1, m):
                    if a[q] >= b[p]:
                        break
                    L = min(b[p], b[q]) - a[q]
                    if L > 0 and abs(idx[p] - idx[q]) > 1:
                        lo, hi_ = a[q], min(b[p], b[q])
                        if horiz:
                            seg = (lo, line[p], hi_, line[p])
                        else:
                            seg = (line[p], lo, line[p], hi_)
                        res.append((min(idx[p], idx[q]), max(idx[p], idx[q]), L) + seg)
            k = m
    return np.array(res, dtype=np.int64).reshape(-1, 7)


# ---------------------------------------------------------------- global measures


def _entropy(counts):
    p = np.array([c for c in counts if c > 0], float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def turn_entropy(turns):
    h1 = _entropy(Counter(turns.tolist()).values())
    pairs = Counter(zip(turns[:-1].tolist(), turns[1:].tolist()))
    prev = Counter(turns[:-1].tolist())
    h2 = 0.0
    for a, na in prev.items():
        cond = [c for (x, _), c in pairs.items() if x == a]
        h2 += na / (len(turns) - 1) * _entropy(cond)
    return h1, h2


def runs(mask):
    """(start, length) of maximal True runs."""
    out, s = [], None
    for k, v in enumerate(list(mask) + [False]):
        if v and s is None:
            s = k
        elif not v and s is not None:
            out.append((s, k - s)); s = None
    return out


def occupied_cells(w):
    cells = set()
    for k in range(w.N):
        x0, y0, x1, y1 = int(w.x0[k]), int(w.y0[k]), int(w.x1[k]), int(w.y1[k])
        if y0 == y1:
            for x in range(min(x0, x1), max(x0, x1) + 1):
                cells.add((x, y0))
        else:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                cells.add((x0, y))
    return cells


def convex_hull_area(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(sum(hull[k][0] * hull[k - 1][1] - hull[k - 1][0] * hull[k][1] for k in range(len(hull))))


def box_dimension(cells):
    """Box-counting dimension of the drawing, fitted over scales 2..64 word units."""
    c = np.array(list(cells))
    c -= c.min(0)
    sizes, counts = [], []
    for s in (2, 4, 8, 16, 32, 64):
        counts.append(len(set(map(tuple, c // s))))
        sizes.append(s)
    slope = np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)[0]
    return float(slope)


def gyration_curve(w, samples=200):
    v = w.vertices().astype(float)
    seglen = w.n.astype(float)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    t = np.linspace(0, cum[-1], samples)
    out = []
    for T in t:
        k = max(1, np.searchsorted(cum, T, "right"))
        pts = v[:k + 1]
        wts = np.concatenate([[0], seglen[:k]]) + 1e-9
        c = (pts * wts[:, None]).sum(0) / wts.sum()
        out.append(math.sqrt((((pts - c) ** 2).sum(1) * wts).sum() / wts.sum()))
    return np.linspace(0, 1, samples), np.array(out)


def global_measures(w, with_geometry=True):
    """All global measures of one walk, as a flat dict."""
    n = w.n.astype(float)
    t = w.turn
    L = n.sum()
    m = {}
    # stride (order-free)
    m["stride_mean"] = float(n.mean())
    m["stride_cv"] = float(n.std() / n.mean())
    # turns (order-free shares; order-dependent pairs)
    m["share_left"] = float((t == 1).mean()); m["share_right"] = float((t == -1).mean())
    m["share_straight"] = float((t == 0).mean())
    m["handedness"] = float(t.sum() / 4)             # net rotation in full turns
    h1, h2 = turn_entropy(t)
    m["entropy_h1"] = h1; m["entropy_h2"] = h2
    nz = t != 0
    a, b = t[:-1], t[1:]
    both = (a != 0) & (b != 0)
    m["recoil"] = float(((a == b) & both).sum() / max(both.sum(), 1))
    m["vacillation"] = float(((a == -b) & both).sum() / max(both.sum(), 1))
    rs = runs(t == 0)
    m["corridor_max"] = float(max((r[1] for r in rs), default=0))
    m["corridor_mean"] = float(np.mean([r[1] for r in rs])) if rs else 0.0
    # geometry
    v = w.vertices()
    m["drift"] = float(np.hypot(*(v[-1] - v[0])) / L)
    if with_geometry:
        cr = crossings(w)
        m["crossings"] = float(len(cr))
        m["crossing_rate"] = float(len(cr) / w.N)
        m["crossing_gap_median"] = float(np.median(cr[:, 3] - cr[:, 2])) if len(cr) else 0.0
        ov = overlaps(w)
        m["palimpsest_words"] = float(ov[:, 2].sum()) if len(ov) else 0.0
        cells = occupied_cells(w)
        hull = convex_hull_area(v)
        m["sprawl"] = float(len(cells) / max(hull, 1))
        m["texture"] = box_dimension(cells)
        _, rg = gyration_curve(w, 60)
        m["gyration_final"] = float(rg[-1] / math.sqrt(L))
        m["extent_w"] = float(v[:, 0].max() - v[:, 0].min()); m["extent_h"] = float(v[:, 1].max() - v[:, 1].min())
    return m


ORDER_DEPENDENT = ["drift", "entropy_h2", "recoil", "vacillation", "corridor_max", "corridor_mean",
                   "crossing_rate", "crossing_gap_median", "palimpsest_words", "sprawl", "texture",
                   "gyration_final"]


def shuffle_baseline(lengths, turns, scores, n=200, seed=7):
    """Measures of n random reorderings of the same sentences."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = rng.permutation(len(lengths))
        rows.append(global_measures(walk2d(lengths[p], turns[p], scores[p])))
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def zscores(m, base):
    out = {}
    for k in ORDER_DEPENDENT:
        mu, sd = float(base[k].mean()), float(base[k].std(ddof=1))
        z = (m[k] - mu) / sd if sd > 0 else 0.0
        # empirical two-sided p from the baseline itself (with +1 smoothing)
        p = (1 + (np.abs(base[k] - mu) >= abs(m[k] - mu)).sum()) / (1 + len(base[k]))
        out[k] = dict(value=m[k], mean=mu, sd=sd, z=z, p=float(p))
    return out


# ---------------------------------------------------------------- local structures


def knots(w, cr, cell=24, k=3, min_sep=3):
    """Crossing hotspots: cells (cell x cell words) with the most crossings, 3x3 smoothed.

    For every knot: centre, crossing count, the sentences involved, and how far
    apart in the book those sentences are (its time span).
    """
    if len(cr) == 0:
        return []
    gx = (cr[:, 0] // cell).astype(int); gy = (cr[:, 1] // cell).astype(int)
    cnt = Counter(zip(gx.tolist(), gy.tolist()))
    smooth = Counter()
    for (x, y), c in cnt.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                smooth[(x + dx, y + dy)] += c
    picks = []
    for (x, y), c in smooth.most_common():
        if all(max(abs(x - px), abs(y - py)) >= min_sep for px, py, _ in picks):
            picks.append((x, y, c))
        if len(picks) == k:
            break
    out = []
    for x, y, c in picks:
        sel = (np.abs(gx - x) <= 1) & (np.abs(gy - y) <= 1)
        sub = cr[sel]
        sents = np.unique(sub[:, 2:4])
        out.append(dict(x=float(sub[:, 0].mean()), y=float(sub[:, 1].mean()), crossings=int(len(sub)),
                        sentences=sents.tolist(), first=int(sents.min()), last=int(sents.max()),
                        gap_median=float(np.median(sub[:, 3] - sub[:, 2]))))
    return out


def hub(w, cr):
    """The most-crossed sentence."""
    c = Counter(cr[:, 2].tolist() + cr[:, 3].tolist())
    i, n = c.most_common(1)[0]
    others = sorted(set(cr[cr[:, 2] == i][:, 3].tolist() + cr[cr[:, 3] == i][:, 2].tolist()))
    return dict(i=int(i), crossings=int(n), crossed_by=others)


def longest_return(cr):
    """The crossing whose two sentences are farthest apart in the book."""
    k = int(np.argmax(cr[:, 3] - cr[:, 2]))
    x, y, i, j = cr[k]
    return dict(x=int(x), y=int(y), i=int(i), j=int(j), gap=int(j - i))


def coil(w, window=12):
    """The window of sentences with the largest net rotation (a spiral)."""
    c = np.concatenate([[0], np.cumsum(w.turn)])
    net = c[window:] - c[:-window]
    k = int(np.argmax(np.abs(net)))
    return dict(first=k, last=k + window - 1, quarter_turns=int(net[k]))


def darkest(w, window=30):
    c = np.concatenate([[0], np.cumsum(w.score)])
    s = c[window:] - c[:-window]
    k = int(np.argmin(s))
    return dict(first=k, last=k + window - 1, score=float(s[k]))


def longest_corridor(w):
    rs = runs(w.turn == 0)
    s, L = max(rs, key=lambda r: (w.n[r[0]:r[0] + r[1]].sum(), r[1]))
    return dict(first=int(s), last=int(s + L - 1), sentences=int(L), words=int(w.n[s:s + L].sum()))


def chapter_overlap(w, chapter_starts):
    """Jaccard overlap of the ground (unit cells) covered by each pair of chapters."""
    bounds = list(chapter_starts) + [w.N]
    sets = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        cells = set()
        for k in range(a, b):
            x0, y0, x1, y1 = int(w.x0[k]), int(w.y0[k]), int(w.x1[k]), int(w.y1[k])
            if y0 == y1:
                cells.update((x, y0) for x in range(min(x0, x1), max(x0, x1) + 1))
            else:
                cells.update((x0, y) for y in range(min(y0, y1), max(y0, y1) + 1))
        sets.append(cells)
    n = len(sets)
    J = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            u = len(sets[p] | sets[q])
            J[p, q] = len(sets[p] & sets[q]) / u if u else 0
    return J
