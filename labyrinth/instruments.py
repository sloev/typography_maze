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
    surf.finish()
    return dict(speech_share=float(m.mean()), longest_silence=int(max((len(x) for x in "".join("1" if v else "0" for v in m).split("1")), default=0)))


# ================================================================== five more instruments
# breath     every clause (the words between two punctuation marks) is a vertical stroke as
#            tall as the clause is long, packed in rows: how long a book holds its breath.
# sky        one mark per sentence, by how it ends: a dot (.), an open ring (?), a bar (!).
# pronouns   a second walk, steered by who is spoken of: I north, you south, he east,
#            she west, we north-east, they south-west; one step per pronoun.
# clock      every mention of a time of day on a 24-hour dial (midnight at the top, noon
#            at the bottom); the distance from the centre is how far into the book it comes.
# light      the balance of light words against dark words through the book, as one
#            1-bit band: white where the book is bright, black where it is dark.

_CLAUSE = re.compile(r"[,;:.!?()—–\"“”]|--")


def clauses(book):
    out = []
    for s in book.sentences:
        for part in _CLAUSE.split(s.text):
            n = len(re.findall(r"[A-Za-zÀ-ÿ'’]+", part))
            if n:
                out.append((n, s.i, part.strip()))
    return out


def breath(book, path, wmm=120.0, row_h=9.0, rows=4, gap=2.5, cap=40):
    cl = clauses(book)
    lens = np.array([c[0] for c in cl], float)
    hmm = rows * row_h + (rows - 1) * gap
    surf, ctx = _surface(path, wmm, hmm)
    per = len(lens) / rows
    bins = int(wmm / 0.22)               # one stroke per 0.22 mm: the longest clause in each bin
    ctx.set_line_width(0.12)
    for r in range(rows):
        seg = lens[int(r * per):int((r + 1) * per)]
        y0 = r * (row_h + gap) + row_h
        edges = np.linspace(0, len(seg), bins + 1).astype(int)
        for b in range(bins):
            if edges[b + 1] <= edges[b]:
                continue
            v = min(seg[edges[b]:edges[b + 1]].max(), cap) / cap
            x = (b + 0.5) * wmm / bins
            ctx.move_to(x, y0); ctx.line_to(x, y0 - v * row_h)
        ctx.stroke()
    surf.finish()
    k = int(np.argmax(lens))
    return dict(clauses=len(cl), mean=float(lens.mean()), longest=int(lens[k]), longest_sentence=cl[k][1],
                long_share=float((lens >= 20).mean()))


def ending(text):
    t = text.rstrip().rstrip("\"”’')]").rstrip()
    return t[-1:] if t else "."


def sky(book, path, wmm=120.0, cell=1.05):
    S = book.sentences
    cols = int(wmm / cell)
    rows = math.ceil(len(S) / cols)
    hmm = rows * cell + 0.5
    surf, ctx = _surface(path, wmm, hmm)
    ends = [ending(s.text) for s in S]
    for k, e in enumerate(ends):
        x = (k % cols + 0.5) * cell; y = (k // cols + 0.5) * cell
        if e == "?":
            ctx.set_line_width(0.12); ctx.arc(x, y, cell * 0.36, 0, 2 * math.pi); ctx.stroke()
        elif e == "!":
            ctx.set_line_width(0.22); ctx.move_to(x, y - cell * 0.42); ctx.line_to(x, y + cell * 0.42); ctx.stroke()
        else:
            ctx.arc(x, y, 0.09, 0, 2 * math.pi); ctx.fill()
    surf.finish()
    q = np.array([e == "?" for e in ends]); x = np.array([e == "!" for e in ends])
    # the densest run of questions: 20 sentences with the most question marks
    c = np.concatenate([[0], np.cumsum(q)])
    win = 20
    k = int(np.argmax(c[win:] - c[:-win])) if len(q) > win else 0
    return dict(questions=float(q.mean()), exclamations=float(x.mean()), height=hmm,
                q_run_start=k, q_run=int(c[k + win] - c[k]) if len(q) > win else int(q.sum()))


PRONOUNS = {"i": (0, 1), "me": (0, 1), "my": (0, 1), "mine": (0, 1), "myself": (0, 1),
            "you": (0, -1), "your": (0, -1), "yours": (0, -1), "yourself": (0, -1),
            "he": (1, 0), "him": (1, 0), "his": (1, 0), "himself": (1, 0),
            "she": (-1, 0), "her": (-1, 0), "hers": (-1, 0), "herself": (-1, 0),
            "we": (0.7071, 0.7071), "us": (0.7071, 0.7071), "our": (0.7071, 0.7071), "ourselves": (0.7071, 0.7071),
            "they": (-0.7071, -0.7071), "them": (-0.7071, -0.7071), "their": (-0.7071, -0.7071),
            "themselves": (-0.7071, -0.7071)}
GROUP = {"i": "I", "me": "I", "my": "I", "mine": "I", "myself": "I", "you": "you", "your": "you",
         "yours": "you", "yourself": "you", "he": "he", "him": "he", "his": "he", "himself": "he",
         "she": "she", "her": "she", "hers": "she", "herself": "she", "we": "we", "us": "we", "our": "we",
         "ourselves": "we", "they": "they", "them": "they", "their": "they", "themselves": "they"}


def pronoun_walk(book, path, wmm=36.0, hmm=36.0):
    from .draw import draw_path
    steps, counts = [], Counter()
    for s in book.sentences:
        for w in s.words:
            k = w.lower().replace("’", "'")
            if k in PRONOUNS:
                steps.append(PRONOUNS[k]); counts[GROUP[k]] += 1
    v = np.vstack([[0, 0], np.cumsum(np.array(steps, float), 0)])
    draw_path(v, path, wmm, hmm, line_mm=0.08, fixed=((v.min(0) + v.max(0)) / 2, max(np.ptp(v[:, 0]), np.ptp(v[:, 1])) * 1.04))
    total = sum(counts.values())
    end = v[-1]
    heading = (math.degrees(math.atan2(end[0], end[1])) + 360) % 360      # compass bearing, 0 = north
    return dict(shares={k: counts[k] / total for k in ("I", "you", "he", "she", "we", "they")},
                bearing=heading, steps=total)


TIMES = {"midnight": 0, "dawn": 5.5, "daybreak": 5.5, "sunrise": 6, "breakfast": 8, "morning": 9, "mornings": 9,
         "forenoon": 10, "noon": 12, "midday": 12, "lunch": 13, "afternoon": 15, "afternoons": 15, "tea": 17,
         "dinner": 18, "sunset": 19, "evening": 19, "evenings": 19, "dusk": 20, "twilight": 20, "supper": 21,
         "tonight": 22, "night": 23, "nights": 23, "nightfall": 21}


def clock(book, path, size=36.0):
    total = sum(len(s.words) for s in book.sentences)
    pos, marks = 0, []
    for s in book.sentences:
        for w in s.words:
            k = w.lower()
            if k in TIMES:
                marks.append((TIMES[k], pos / total, k))
            pos += 1
    R = size / 2 - 0.8
    surf, ctx = _surface(path, size, size)
    ctx.translate(size / 2, size / 2)
    ctx.set_line_width(0.15); ctx.arc(0, 0, R, 0, 2 * math.pi); ctx.stroke()
    for h in range(24):                      # hour ticks; longer at midnight, 6, noon and 18
        a = h / 24 * 2 * math.pi - math.pi / 2
        L = 1.6 if h % 6 == 0 else 0.6
        ctx.set_line_width(0.3 if h % 6 == 0 else 0.12)
        ctx.move_to(R * math.cos(a), R * math.sin(a)); ctx.line_to((R - L) * math.cos(a), (R - L) * math.sin(a))
        ctx.stroke()
    rng = np.random.default_rng(1)
    for h, p, _ in marks:
        a = (h + rng.uniform(-0.35, 0.35)) / 24 * 2 * math.pi - math.pi / 2
        r = (0.12 + 0.82 * p) * R
        ctx.arc(r * math.cos(a), r * math.sin(a), 0.38, 0, 2 * math.pi); ctx.fill()
    surf.finish()
    hours = np.array([m[0] for m in marks]) if marks else np.zeros(0)
    night = float(np.mean((hours >= 21) | (hours < 5))) if len(hours) else 0.0
    ang = hours / 24 * 2 * math.pi
    mean_h = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 15) % 24 if len(hours) else 0
    return dict(mentions=len(marks), night=night, mean_hour=float(mean_h),
                top=Counter(m[2] for m in marks).most_common(3))


LIGHT = {"light", "lights", "bright", "brightly", "brightness", "sun", "sunlight", "sunshine", "sunny", "shine",
         "shining", "shone", "glow", "glowing", "gleam", "gleaming", "glitter", "glittering", "lamp", "lamps",
         "candle", "candles", "daylight", "radiant", "brilliant", "illuminated", "lit", "white", "fire", "flame",
         "sparkling", "dazzling", "luminous", "blaze", "blazing"}
DARK = {"dark", "darker", "darkest", "darkness", "darkened", "black", "shadow", "shadows", "shadowy", "gloom",
        "gloomy", "dim", "dimly", "dusk", "obscure", "obscurity", "murky", "fog", "foggy", "mist", "misty",
        "blackness", "shade", "shady", "night", "midnight", "unlit", "dull", "grey", "gray", "smoke", "smoky"}


def light_meter(book, out_png, wmm=120.0, hmm=7.0, window=3000):
    words = [w.lower() for s in book.sentences for w in s.words]
    L = np.array([w in LIGHT for w in words], float); D = np.array([w in DARK for w in words], float)
    k = np.ones(window)
    l = np.convolve(L, k, "same"); d = np.convolve(D, k, "same")
    bal = np.where(l + d > 0, (l - d) / np.maximum(l + d, 1), 0)     # +1 all light .. -1 all dark
    tone = 0.5 + 0.5 * bal                                          # 1 white .. 0 black
    tone = np.where(l + d >= 3, tone, 1.0)        # too few light or dark words nearby: left blank
    W = 1200
    xs = np.linspace(0, len(tone) - 1, W).astype(int)
    img = np.tile(tone[xs], (60, 1))
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import halftone
    ink = halftone.screen_array(img, width_in=wmm / 25.4, kind="dots", seed=3)
    from PIL import Image
    Image.fromarray(~ink).convert("1").save(out_png, dpi=(1200, 1200))
    valid = (l + d) >= 3
    k_dark = int(np.argmin(np.where(valid, bal, 2))); k_light = int(np.argmax(np.where(valid, bal, -2)))
    return dict(light=int(L.sum()), dark=int(D.sum()), balance=float((L.sum() - D.sum()) / max(L.sum() + D.sum(), 1)),
                darkest_at=k_dark / len(words), lightest_at=k_light / len(words), covered=float(valid.mean()))
