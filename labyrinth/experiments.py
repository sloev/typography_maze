"""Turning-rule experiments, each shown from every viewpoint.

    python3 -m labyrinth.experiments books/the-trial.yaml

For each rule in labyrinth.variants (quarter, graded, rolling) writes to
build/<slug>/variants/:
    <rule>-plan.pdf        2D, from above
    <rule>-shaft.pdf       the mineshaft, from an angled side view
    <rule>-stand.png       standing letters at eye level (1-bit)
    <rule>-high.png        standing letters from above (1-bit)
    numbers.tex            statistics per rule for the LaTeX chapter
and a contact sheet, contact.png.
"""
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image

from . import draw, letters, text, variants
from .build import Numbers, fmt, quote, ROOT

PLAN_W, PLAN_H = 57.0, 80.0      # half a text block: two views side by side
SHAFT_W, SHAFT_H = 57.0, 80.0


def stats(p, chapter_starts):
    v = p.vertices2d()
    v3, pitch = p.vertices3d()
    L = p.length.sum()
    return dict(
        strokes=len(p.length),
        mean_turn=float(np.abs(p.turn_deg).mean()),
        net_turns=float(p.turn_deg.sum() / 360),
        drift=float(np.hypot(*(v[-1] - v[0])) / L),
        depth=float(-v3[:, 2].min()),
        final_pitch=float(pitch[-1]),
        min_pitch=float(pitch.min()),
        max_pitch=float(pitch.max()),
        deepest=int(np.argmin(v3[1:, 2])),
    )


def main(cfg_path, views=("plan", "shaft", "stand", "high"), W=2000, H=1333):
    cfg = yaml.safe_load(open(cfg_path))
    from .build import load_book
    book = load_book(cfg)
    out = os.path.join(ROOT, "build", cfg["slug"], "variants")
    os.makedirs(out, exist_ok=True)
    N = Numbers()
    N["window"] = str(variants.WINDOW); N["gain"] = fmt(variants.GAIN, 0); N["delta"] = fmt(variants.DELTA, 2)
    for rule in variants.RULES:
        p = variants.build(book, rule)
        st = stats(p, book.chapter_starts)
        for k, v in st.items():
            if k != "deepest":
                N[f"{rule}-{k}"] = fmt(v, 2 if abs(v) < 10 else 1)
        g = int(p.group[st["deepest"]])
        N[f"{rule}-deepest"] = f"\\S\\,{g + 1}"; N[f"{rule}-deepest-ch"] = str(book.sentences[g].chapter + 1)
        N[f"{rule}-deepest-quote"] = quote(book.sentences[g].text, 24)
        # stroke index at each chapter start
        first_stroke = {int(gr): k for k, gr in reversed(list(enumerate(p.group)))}
        marks = [first_stroke[s] for s in book.chapter_starts]
        if "plan" in views:
            draw.draw_path(p.vertices2d(), os.path.join(out, f"{rule}-plan.pdf"), PLAN_W, PLAN_H, marks=marks)
        if "shaft" in views:
            v3, _ = p.vertices3d()
            step = 4 if len(v3) > 20000 else 1      # rolling: join words into runs of 4 so the line is continuous
            keep = np.unique(np.r_[np.arange(0, len(v3), step), len(v3) - 1])
            remap = {k: int(np.searchsorted(keep, k)) for k in marks}
            v3 = v3[keep]
            _, vs = draw.draw_mineshaft(v3, os.path.join(out, f"{rule}-shaft.pdf"), SHAFT_W, SHAFT_H,
                                        azimuth=35, elevation=24, lw_near=0.3, lw_far=0.08,
                                        marks=[remap[k] for k in marks])
            N[f"{rule}-vscale"] = f"1\\,:\\,{1 / vs:.0f}" if vs < 1 else "1\\,:\\,1"
        ps = p.with_lengths(letters.set_widths(p.text, add_space=(rule == "rolling")))
        for view in ("stand", "high"):
            if view in views:
                info = letters.render(ps, os.path.join(out, f"{rule}-{view}.png"), view, W=W, H=H,
                                      scene_radius=60 if rule != "rolling" else 45)
                N[f"{rule}-{view}-near"] = str(info["sentences_near"])
                N[f"{rule}-{view}-ref"] = f"\\S\\,{info['base_group'] + 1}"
        print(rule, {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})
    N.write(os.path.join(out, "numbers.tex"))
    contact(out, views)


def contact(out, views, cell=(420, 300)):
    import subprocess
    rows = []
    for rule in variants.RULES:
        row = []
        for view in views:
            f = os.path.join(out, f"{rule}-{view}")
            if os.path.exists(f + ".pdf"):
                subprocess.run(["pdftoppm", "-r", "90", "-png", "-singlefile", f + ".pdf", f + "-preview"], check=True)
                im = Image.open(f + "-preview.png")
            elif os.path.exists(f + "_tone.png"):
                im = Image.open(f + "_tone.png")
            else:
                im = Image.new("L", cell, 255)
            im = im.convert("L"); im.thumbnail(cell, Image.LANCZOS)
            c = Image.new("L", cell, 255); c.paste(im, ((cell[0] - im.width) // 2, (cell[1] - im.height) // 2))
            row.append(c)
        rows.append(row)
    sheet = Image.new("L", (len(views) * (cell[0] + 8), len(rows) * (cell[1] + 8)), 120)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im, (c * (cell[0] + 8), r * (cell[1] + 8)))
    sheet.save(os.path.join(out, "contact.png"))


if __name__ == "__main__":
    main(sys.argv[1])
