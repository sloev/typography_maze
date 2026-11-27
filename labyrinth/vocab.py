"""The vocabulary map: a book's words as black pixels on a map of the dictionary.

Every pixel of a disc is one dictionary word. The words are laid out like the
groove of a record: an Archimedean spiral that starts at the centre with the
most common word and winds outward one pixel per turn, so that the rings touch
and every pixel is used exactly once. The further out, the rarer the word. A
book turns the pixel of every word it uses black. The result is a picture of
the book's vocabulary: a dense centre of everyday words, thinning towards the
rim as the book reaches for rarer ones.

Dictionary: the lowercase, apostrophe-free entries of the system word list
(SCOWL "american-english", /usr/share/dict/words). Commonness: the Zipf
frequency of each word in the `wordfreq` English data (Speer, 2022), used here
as a stand-in for difficulty, since a rarer word is, on average, a harder one.
Words of equal frequency are ordered alphabetically.
"""
import math
import os
import re

import numpy as np
from PIL import Image

from . import sentiment
from .text import CACHE

DICT_FILE = "/usr/share/dict/words"


def dictionary():
    """(words ordered common -> rare, their Zipf frequencies)."""
    cache = os.path.join(CACHE, "vocab-order.txt")
    if os.path.exists(cache):
        rows = [l.rstrip("\n").split("\t") for l in open(cache, encoding="utf-8")]
        return [w for w, _ in rows], np.array([float(z) for _, z in rows])
    from wordfreq import zipf_frequency
    if not os.path.exists(DICT_FILE):
        raise RuntimeError("install a system word list (apt-get install wamerican)")
    words = sorted({w.strip() for w in open(DICT_FILE, encoding="utf-8")
                    if re.fullmatch(r"[a-z]+", w.strip())})
    z = [zipf_frequency(w, "en") for w in words]
    order = sorted(range(len(words)), key=lambda k: (-z[k], words[k]))
    words = [words[k] for k in order]; z = [z[k] for k in order]
    with open(cache, "w", encoding="utf-8") as f:
        for w, q in zip(words, z):
            f.write(f"{w}\t{q:.2f}\n")
    return words, np.array(z)


def diagonal_cells(side):
    """Grid cells ordered along anti-diagonals from the top-left to the bottom-right."""
    cells = [(r, c) for r in range(side) for c in range(side)]
    cells.sort(key=lambda rc: (rc[0] + rc[1], rc[0]))
    return cells


def spiral_cells(n):
    """n grid cells ordered along an Archimedean spiral of pitch 1 pixel (r = turns),
    starting at the centre. Returns (cells as (row, col), side of the square image)."""
    R = math.sqrt(n / math.pi) + 2
    side = 2 * math.ceil(R) + 1
    c0 = side // 2
    yy, xx = np.mgrid[0:side, 0:side]
    x = xx - c0 + 0.0; y = c0 - yy + 0.0
    r = np.hypot(x, y)
    th = (np.arctan2(y, x) % (2 * math.pi)) / (2 * math.pi)       # 0..1 of a turn
    k = np.round(r - th)                                          # which ring of the groove
    pos = k + th                                                  # position along the groove, in turns
    order = np.lexsort((r.ravel(), pos.ravel()))[:n]
    return [(int(i // side), int(i % side)) for i in order], side


def book_words(book):
    """The distinct dictionary-form words a book uses (lowercased; names excluded)."""
    names = sentiment.book_names([s.text for s in book.sentences])
    used = {}
    for s in book.sentences:
        for tok, w, start in sentiment._tokens(s.text):
            if not w or not re.fullmatch(r"[A-Za-z]+", w):
                continue
            if w[:1].isupper() and not start and w != "I":
                continue                      # a name or title, not vocabulary
            k = w.lower()
            if k in names:
                continue
            used[k] = used.get(k, 0) + 1
    return used


def vocabulary_map(book, out_png, scale=4):
    words, z = dictionary()
    rank = {w: k for k, w in enumerate(words)}
    cells, side = spiral_cells(len(words))
    used = book_words(book)
    img = np.full((side, side), 255, np.uint8)
    ranks = []
    for w in used:
        k = rank.get(w)
        if k is not None:
            r, c = cells[k]
            img[r, c] = 0
            ranks.append(k)
    ranks = np.array(sorted(ranks))
    # the unused cells after the last word stay white; mark the grid edge so the square reads
    Image.fromarray(img).resize((side * scale, side * scale), Image.NEAREST).convert("1").save(out_png)
    N = len(words)
    tokens = sum(used.values())
    in_dict = sum(c for w, c in used.items() if w in rank)
    return dict(
        dictionary=N, side=side,
        types=len(used), types_in_dict=len(ranks), token_coverage=in_dict / max(tokens, 1),
        median_rank=float(np.median(ranks)) / N,          # 0 = most common, 1 = rarest
        rare_share=float((ranks > N / 2).mean()),         # share of its words in the rarer half
        reach=float(np.percentile(ranks, 95)) / N,        # how far into the rare words it reaches
        rarest=[words[k] for k in ranks[-6:][::-1]],
    )


def sampled_types(book, size=6000, samples=20, seed=5):
    """Distinct dictionary words in a stretch of `size` running words, averaged over
    random stretches, so books of very different lengths can be compared fairly."""
    words, _ = dictionary()
    known = set(words)
    names = sentiment.book_names([s.text for s in book.sentences])
    seq = []
    for s in book.sentences:
        for tok, w, start in sentiment._tokens(s.text):
            if w and re.fullmatch(r"[A-Za-z]+", w) and not (w[:1].isupper() and not start and w != "I") \
                    and w.lower() not in names:
                seq.append(w.lower())
    if len(seq) <= size:
        return float(len({w for w in seq if w in known}))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(seq) - size, samples)
    return float(np.mean([len({w for w in seq[a:a + size] if w in known}) for a in starts]))
