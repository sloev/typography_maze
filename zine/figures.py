"""Figures that only the zine needs.

    python3 zine/figures.py        (from the repository root, after `make`)

For every book: a full-page maze, the graded and rolling walks (plan), and a
standing-letters view from above. For The Trial also the 'how it works' walk.
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from labyrinth import draw, letters, sentiment, variants, walk     # noqa: E402
from labyrinth.build import load_book                              # noqa: E402

OUT = os.path.join(ROOT, "build", "zine")
os.makedirs(OUT, exist_ok=True)
SLUGS = ["the-trial", "bartleby", "yellow-wallpaper", "heart-of-darkness", "pride-and-prejudice"]
only = sys.argv[1:] or SLUGS

for slug in only:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "books", f"{slug}.yaml")))
    book = load_book(cfg)
    S = book.sentences
    names = sentiment.book_names([s.text for s in S])
    sc = np.array([sentiment.score_text(s.text, names) for s in S], float)
    t = np.array([sentiment.turn(x) for x in sc])
    n = np.array([len(s.words) for s in S])
    w = walk.walk2d(n, t, sc)
    # full page, hairline, no marks
    draw.draw_path(w.vertices(), os.path.join(OUT, f"maze-{slug}.pdf"), 128, 176, line_mm=0.08)
    # the three turning rules, as plans for the atlas and as full pages
    for rule in variants.RULES:
        p = variants.build(book, rule)
        draw.draw_path(p.vertices2d(), os.path.join(OUT, f"plan-{rule}-{slug}.pdf"), 40, 40, line_mm=0.06)
        if rule != "quarter":
            draw.draw_path(p.vertices2d(), os.path.join(OUT, f"maze-{rule}-{slug}.pdf"), 128, 176, line_mm=0.08)
    # standing letters from above, quarter turns (The Trial's come from the experiments)
    if slug != "the-trial" or not os.path.exists(os.path.join(ROOT, "build", "the-trial", "variants", "quarter-high.png")):
        p = variants.build(book, "quarter")
        ps = p.with_lengths(letters.set_widths(p.text))
        letters.render(ps, os.path.join(OUT, f"letters-{slug}.png"), "high", W=1800, H=1200,
                       scene_radius=60, width_in=128 / 25.4)
    print("figures:", slug)

if "the-trial" in only:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "books", "the-trial.yaml")))
    book = load_book(cfg)
    S = book.sentences
    names = sentiment.book_names([s.text for s in S])
    sc = np.array([sentiment.score_text(s.text, names) for s in S], float)
    t = np.array([sentiment.turn(x) for x in sc])
    n = np.array([len(s.words) for s in S])
    w = walk.walk2d(n, t, sc)
    draw.draw_path(w.vertices(), os.path.join(OUT, "cover-maze.pdf"), 120, 168, line_mm=0.09)
    # how it works: a short passage from chapter 1 that turns often and stays compact
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
