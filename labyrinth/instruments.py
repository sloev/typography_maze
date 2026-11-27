"""Four more instruments for looking at a book, all drawn in pure black on white.

tree_rings  The book as the cross-section of a trunk. The centre is the first page.
            Each chapter is a ring whose area is proportional to its length in
            words. Inside a ring, every sentence is a radial tick at its place
            around the circle (clockwise from the top, in reading order), and the
            tick is as long as the sentence's words are long: a ring of short,
            plain words is pale and thin-grained, a ring of long words is dark.
            Books without chapters are cut into ten rings of equal length.

loom        The cast as woven cloth. The most-named people and places are the warp
            threads, one column each, in order of first appearance. The book is
            the weft, read from top to bottom in rows of equal length. A cell is
            black where that name occurs in that stretch of the book.

echoes      The book remembering itself. Every word that recurs (not the commonest
            words, and not names) gets one half circle above a line that runs the
            length of the book, spanning its longest absence: from the place the
            book last uses it to the place it comes back. Low arches are words
            never long forgotten; high arches are words that return from far away.

tide        Speech and silence. The share of words spoken inside quotation marks,
            in a moving window, drawn as a filled black tide over the length of
            the book and folded into rows like a seismograph printout.
"""
import math
import re
from collections import Counter, defaultdict

import cairo
import numpy as np

from . import sentiment
from .draw import MM, _surface

NOT_NAMES = {"I", "Mr", "Mrs", "Miss", "Ms", "Sir", "Lady", "Lord", "Dr", "God", "Madam", "Madame",
             "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Christmas",
             "Oh", "O", "Ah", "No", "Yes", "Well", "But", "And", "The", "Herr", "Frau"}


def _book_tokens(book):
    """[(sentence index, word, is-start)] over the whole book."""
    out = []
    for s in book.sentences:
        for tok, w, start in sentiment._tokens(s.text):
            if w:
                out.append((s.i, w, start))
    return out


# ------------------------------------------------------------------ tree rings
def tree_rings(book, path, size=56.0, rings=None):
    S = book.sentences
    n = np.array([len(s.words) for s in S], float)
    wl = np.array([np.mean([len(w) for w in s.words]) for s in S])
    if rings is None:
        if len(book.chapter_starts) > 1:
            bounds = list(book.chapter_starts) + [len(S)]
        else:
            cum = np.cumsum(n) / n.sum()
            bounds = [0] + [int(np.searchsorted(cum, q)) for q in np.linspace(0.1, 0.9, 9)] + [len(S)]
    total = n.sum()
    R = size / 2 - 0.6
    lo_wl, hi_wl = np.percentile(wl, 2), np.percentile(wl, 98)
    surf, ctx = _surface(path, size, size)
    ctx.translate(size / 2, size / 2)
    acc = 0.0
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for a, b in zip(bounds[:-1], bounds[1:]):
        r0 = R * math.sqrt(acc / total)
        acc += n[a:b].sum()
        r1 = R * math.sqrt(acc / total)
        seg = n[a:b]; pos = (np.cumsum(seg) - seg / 2) / max(seg.sum(), 1)
        ctx.set_line_width(max(0.025, min(0.18, 2 * math.pi * r1 / max(len(seg), 1) * 0.45)))
        for p, q in zip(pos, wl[a:b]):
            f = np.clip((q - lo_wl) / (hi_wl - lo_wl), 0, 1)
            th = 2 * math.pi * p - math.pi / 2
            rr = r0 + (r1 - r0) * (0.12 + 0.8 * f)
            ctx.move_to(r0 * math.cos(th), r0 * math.sin(th)); ctx.line_to(rr * math.cos(th), rr * math.sin(th))
            ctx.stroke()
        ctx.set_line_width(0.22 if len(bounds) <= 13 else 0.07)
        ctx.arc(0, 0, r1, 0, 2 * math.pi); ctx.stroke()
    surf.finish()
    return dict(rings=len(bounds) - 1, mean_word_length=float((wl * n).sum() / total))


# ------------------------------------------------------------------ loom
def cast(book, top=12, min_count=8):
    """The most-named people and places, in order of first appearance."""
    counts, first = Counter(), {}
    for si, w, start in _book_tokens(book):
        w = re.sub(r"[’']s$", "", w)          # Kurtz's -> Kurtz
        if start or not w[:1].isupper() or w in NOT_NAMES or (len(w) > 1 and w.isupper()) or w.startswith(("I'", "I’")):
            continue
        counts[w] += 1
        first.setdefault(w, si)
    names = [w for w, c in counts.most_common(top * 3) if c >= min_count]
    # drop words that are mostly lowercase in the book (ordinary words capitalised in titles)
    lower = Counter(w for _, w, _ in _book_tokens(book) if w[:1].islower())
    names = [w for w in names if lower[w.lower()] < counts[w] / 2][:top]
    return sorted(names, key=lambda w: first[w]), counts


def loom(book, path, wmm=120.0, hmm=150.0, rows=150, top=12):
    names, counts = cast(book, top)
    S = book.sentences
    words_per_row = sum(len(s.words) for s in S) / rows
    grid = np.zeros((rows, len(names)), bool)
    pos = 0
    col = {w: k for k, w in enumerate(names)}
    for s in S:
        r = min(rows - 1, int(pos / words_per_row))
        for tok, w, start in sentiment._tokens(s.text):
            w = re.sub(r"[’']s$", "", w)
            if w in col and not start:
                grid[r, col[w]] = True
        pos += len(s.words)
    surf, ctx = _surface(path, wmm, hmm)
    cw, rh = wmm / max(len(names), 1), hmm / rows
    for c in range(len(names)):        # warp threads: a hairline down every column
        ctx.set_line_width(0.08)
        ctx.move_to((c + 0.5) * cw, 0); ctx.line_to((c + 0.5) * cw, hmm); ctx.stroke()
    for r in range(rows):              # weft: a black float wherever the name appears
        for c in range(len(names)):
            if grid[r, c]:
                ctx.rectangle(c * cw + cw * 0.12, r * rh + rh * 0.08, cw * 0.76, rh * 0.84)
        ctx.fill()
    surf.finish()
    together = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            both = (grid[:, a] & grid[:, b]).sum()
            either = (grid[:, a] | grid[:, b]).sum()
            together[(names[a], names[b])] = both / either if either else 0
    pair = max(together, key=together.get) if together else None
    return dict(names=names, counts={w: counts[w] for w in names}, presence={w: float(grid[:, k].mean()) for w, k in col.items()},
                closest=pair, closest_share=together.get(pair, 0))


# ------------------------------------------------------------------ echoes
def echoes(book, path, wmm=120.0, hmm=62.0, min_count=6, max_count=60, max_zipf=5.0, min_span=0.02):
    from wordfreq import zipf_frequency
    toks = [(si, w) for si, w, start in _book_tokens(book)]
    idx = defaultdict(list)
    names = sentiment.book_names([s.text for s in book.sentences])
    for k, (si, w) in enumerate(toks):
        if len(w) < 4 or not w.isalpha():
            continue
        if w[:1].isupper():
            continue
        idx[w.lower()].append(k)
    T = len(toks)
    words = [w for w, ks in idx.items() if min_count <= len(ks) <= max_count
             and w not in names and zipf_frequency(w, "en") < max_zipf]
    surf, ctx = _surface(path, wmm, hmm)
    base = hmm - 0.8
    ctx.set_line_width(0.06)
    spans = []
    for w in words:
        ks = idx[w]
        gaps = [(b - a, a, b) for a, b in zip(ks[:-1], ks[1:])]
        for g, _, _ in gaps:
            spans.append((g, w))
        g, a, b = max(gaps)                  # one arch per word: its longest absence
        if g < min_span * T:
            continue
        x0, x1 = a / T * wmm, b / T * wmm
        ctx.new_sub_path()
        ctx.arc((x0 + x1) / 2, base, (x1 - x0) / 2, math.pi, 2 * math.pi)
    ctx.stroke()
    ctx.set_line_width(0.3); ctx.move_to(0, base); ctx.line_to(wmm, base); ctx.stroke()
    surf.finish()
    spans.sort(reverse=True)
    longest = []
    for g, w in spans:
        if w not in longest:
            longest.append(w)
        if len(longest) == 5:
            break
    return dict(words=len(words), arcs=len(spans), longest=longest,
                far_share=float(np.mean([g > T / 4 for g, _ in spans])) if spans else 0.0)


# ------------------------------------------------------------------ tide
def speech_mask(book):
    """Per word of the book: True if it is spoken inside quotation marks."""
    out = []
    last_par = None
    inside = False
    for s in book.sentences:
        if s.paragraph != last_par:
            inside = False          # a quotation never runs on past a paragraph unseen
            last_par = s.paragraph
        for tok in s.text.split():
            for ch in tok:
                if ch == "“":
                    inside = True
                elif ch == "”":
                    pass
                elif ch == '"':
                    inside = not inside
            spoken = inside or tok.startswith(("“", '"'))
            if re.search(r"[A-Za-z]", tok):
                out.append(spoken)
            if "”" in tok or (tok.endswith('"') and not tok.startswith('"')) or tok.endswith(('",', '."', '?"', '!"')):
                inside = False
    return np.array(out, bool)


def tide(book, path, wmm=120.0, row_h=11.0, rows=6, window=300, gap=3.0):
    m = speech_mask(book).astype(float)
    k = np.ones(window) / window
    level = np.convolve(m, k, mode="same")
    hmm = rows * row_h + (rows - 1) * gap
    surf, ctx = _surface(path, wmm, hmm)
    per = len(level) / rows
    for r in range(rows):
        seg = level[int(r * per):int((r + 1) * per)]
        y0 = r * (row_h + gap) + row_h
        xs = np.linspace(0, wmm, len(seg))
        ctx.move_to(0, y0)
        for x, v in zip(xs[::3], seg[::3]):
            ctx.line_to(x, y0 - v * row_h)
        ctx.line_to(wmm, y0); ctx.close_path(); ctx.fill()
        ctx.set_line_width(0.15); ctx.move_to(0, y0); ctx.line_to(wmm, y0); ctx.stroke()
        ctx.set_line_width(0.08); ctx.move_to(0, y0 - row_h); ctx.line_to(wmm, y0 - row_h); ctx.stroke()
    surf.finish()
    return dict(speech_share=float(m.mean()), longest_silence=int(max((len(x) for x in "".join("1" if v else "0" for v in m).split("1")), default=0)))
