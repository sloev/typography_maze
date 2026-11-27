"""Figures that only the zine needs: the cover maze and the 'how it works' walk.

    python3 zine/figures.py        (from the repository root, after `make`)
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from labyrinth import draw, sentiment, walk          # noqa: E402
from labyrinth.build import load_book                # noqa: E402

OUT = os.path.join(ROOT, "build", "zine")
os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(os.path.join(ROOT, "books", "the-trial.yaml")))
book = load_book(cfg)
S = book.sentences
names = sentiment.book_names([s.text for s in S])
sc = np.array([sentiment.score_text(s.text, names) for s in S], float)
t = np.array([sentiment.turn(x) for x in sc])
n = np.array([len(s.words) for s in S])
w = walk.walk2d(n, t, sc)

# cover: the whole book, hairline, no marks
draw.draw_path(w.vertices(), os.path.join(OUT, "cover-maze.pdf"), 104, 150, line_mm=0.09)

# how it works: the first sentences, large, with their words along the path
# how it works: a short passage from chapter 1 that turns often and stays compact,
# so its sentences can be read along the path at a legible size
K = 8
best = None
for a in range(0, book.chapter_starts[1] - K):
    tt = t[a:a + K]
    if (tt != 0).sum() < 5 or n[a:a + K].max() > 16:
        continue
    vv = walk.walk2d(n[a:a + K], tt).vertices()
    ext = (vv.max(0) - vv.min(0)).max()
    if best is None or ext < best[0]:
        best = (ext, a)
a = best[1]
sub = walk.walk2d(n[a:a + K], t[a:a + K], sc[a:a + K])
v = sub.vertices().astype(float)
cx, cy = (v.min(0) + v.max(0)) / 2
half = (v.max(0) - v.min(0)).max() / 2 + 2.5
draw.draw_cutout(sub, [s.text for s in S[a:a + K]], os.path.join(OUT, "first-steps.pdf"), 96, 96,
                 cx, cy, half, focus=range(K), max_font_mm=3.2)
with open(os.path.join(OUT, "first-steps.tex"), "w") as f:
    f.write(f"\\def\\stepsfrom{{\\S\\,{a + 1}}}\\def\\stepsto{{\\S\\,{a + K}}}\n")
print("zine figures ->", OUT, "| passage", a + 1, "-", a + K, "turns", [int(x) for x in t[a:a + K]])
for i in range(a, a + K):
    print(f"  {int(t[i]):+d} ({int(sc[i]):+d})  {S[i].text}")
