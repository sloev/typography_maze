"""Compare the books: vocabulary maps, drawings side by side, and the numbers.

    python3 -m labyrinth.compare books/*.yaml      (after labyrinth.build for each)

Writes build/compare/:
    vocab-<slug>.png        the vocabulary map of each book
    shape-<slug>.pdf        each drawing fitted to the same frame (shape)
    scale-<slug>.pdf        each drawing at one common scale (size)
    numbers.tex             \\lab{c-<slug>-<key>} for the comparison chapter
"""
import json
import os
import sys

import numpy as np
import yaml

from . import draw, instruments, sentiment, vocab, walk
from .build import ROOT, Numbers, fmt, load_book, tex_escape


def main(paths):
    out = os.path.join(ROOT, "build", "compare")
    os.makedirs(out, exist_ok=True)
    N = Numbers()
    books = []
    for p in paths:
        cfg = yaml.safe_load(open(p))
        book = load_book(cfg)
        S = book.sentences
        names = sentiment.book_names([s.text for s in S])
        sc = np.array([sentiment.score_text(s.text, names) for s in S], float)
        t = np.array([sentiment.turn(x) for x in sc])
        n = np.array([len(s.words) for s in S])
        w = walk.walk2d(n, t, sc)
        a = json.load(open(os.path.join(ROOT, "build", cfg["slug"], "analysis.json")))
        books.append((cfg, book, w, a, n, t))
    # common scale: the largest drawing fills its frame; the others keep its mm-per-word
    ext = max(max(np.ptp(b[2].vertices()[:, 0]), np.ptp(b[2].vertices()[:, 1])) for b in books)
    for cfg, book, w, a, n, t in books:
        slug = cfg["slug"]
        k = f"c-{slug}"
        N[f"{k}-title"] = tex_escape(cfg["title"]); N[f"{k}-author"] = tex_escape(cfg["author"])
        N[f"{k}-year"] = str(cfg.get("year", ""))
        N[f"{k}-sentences"] = fmt(len(book.sentences)); N[f"{k}-words"] = fmt(int(n.sum()))
        N[f"{k}-stride"] = fmt(float(n.mean()), 1)
        m, z = a["measures"], a["z"]
        N[f"{k}-left"] = fmt(100 * m["share_left"], 0); N[f"{k}-right"] = fmt(100 * m["share_right"], 0)
        N[f"{k}-straight"] = fmt(100 * m["share_straight"], 0)
        for key in ("recoil", "entropy_h2", "corridor_mean", "drift", "crossing_rate", "gyration_final"):
            N[f"{k}-z-{key}"] = fmt(z[key]["z"], 1)
            N[f"{k}-sig-{key}"] = "1" if abs(z[key]["z"]) >= 3 else "0"
        N[f"{k}-crossings"] = fmt(m["crossing_rate"], 2)
        N[f"{k}-net"] = fmt(int(t.sum()))
        names = sentiment.book_names([s.text for s in book.sentences])
        t_r = np.array([sentiment.turn(sentiment.score_text(s.text, names | {"like"})) for s in book.sentences])
        N[f"{k}-net-r"] = fmt(int(t_r.sum()))
        N[f"{k}-names"] = ", ".join(sorted(sentiment.book_names([s.text for s in book.sentences]))) or "none"
        # vocabulary
        info = vocab.vocabulary_map(book, os.path.join(out, f"vocab-{slug}.png"))
        N[f"{k}-types"] = fmt(info["types_in_dict"]); N[f"{k}-median"] = fmt(100 * info["median_rank"], 1)
        N[f"{k}-reach"] = fmt(100 * info["reach"], 0); N[f"{k}-rare"] = fmt(100 * info["rare_share"], 1)
        N[f"{k}-rarest"] = ", ".join(f"\\emph{{{x}}}" for x in info["rarest"][:4])
        # per thousand words of text: new dictionary words (so short and long books compare)
        N[f"{k}-types-per-k"] = fmt(1000 * info["types_in_dict"] / n.sum(), 0)
        N[f"{k}-types-6k"] = fmt(int(round(vocab.sampled_types(book, 6000))))
        # drawings
        draw.draw_path(w.vertices(), os.path.join(out, f"shape-{slug}.pdf"), 36, 36, line_mm=0.08)
        v = w.vertices().astype(float)
        c = (v.min(0) + v.max(0)) / 2
        draw.draw_path(v, os.path.join(out, f"scale-{slug}.pdf"), 36, 36, line_mm=0.06, fixed=(c, ext))
        # the other instruments
        r = instruments.tree_rings(book, os.path.join(out, f"rings-{slug}.pdf"), size=36)
        N[f"{k}-rings"] = str(r["rings"]); N[f"{k}-wordlen"] = fmt(r["mean_word_length"], 2)
        lm = instruments.loom(book, os.path.join(out, f"loom-{slug}.pdf"), wmm=22, hmm=110, rows=110, top=10)
        N[f"{k}-cast"] = ", ".join(tex_escape(x) for x in lm["names"])
        if lm["closest"]:
            N[f"{k}-pair"] = " and ".join(tex_escape(x) for x in lm["closest"])
            N[f"{k}-pair-share"] = fmt(100 * lm["closest_share"], 0)
        top = max(lm["presence"], key=lm["presence"].get)
        N[f"{k}-lead"] = tex_escape(top); N[f"{k}-lead-share"] = fmt(100 * lm["presence"][top], 0)
        ec = instruments.echoes(book, os.path.join(out, f"echo-{slug}.pdf"), wmm=120, hmm=26)
        N[f"{k}-echo-words"] = fmt(ec["words"]); N[f"{k}-echo-far"] = fmt(100 * ec["far_share"], 1)
        N[f"{k}-echo-longest"] = ", ".join(f"\\emph{{{x}}}" for x in ec["longest"][:3])
        td = instruments.tide(book, os.path.join(out, f"tide-{slug}.pdf"), wmm=120, row_h=7, rows=3, gap=2)
        N[f"{k}-speech"] = fmt(100 * td["speech_share"], 0); N[f"{k}-silence"] = fmt(td["longest_silence"])
        json.dump(dict(vocab=info, loom=lm, echoes=ec, tide=td, rings=r),
                  open(os.path.join(out, f"{slug}.json"), "w"), indent=1, default=str)
        print(slug, info["types_in_dict"], "words in the dictionary,", f"median rank {info['median_rank']:.3f}",
              "| cast", lm["names"][:5], "| speech", round(td["speech_share"], 2), "| far echoes", round(ec["far_share"], 3))
    words, _ = vocab.dictionary()
    cells, side = vocab.spiral_cells(len(words))
    N["vocab-rim"] = f"{(len(words) / np.pi) ** 0.5 / side:.4f}"   # disc radius as a fraction of the image
    N["vocab-words"] = fmt(len(words))
    N.write(os.path.join(out, "numbers.tex"))


if __name__ == "__main__":
    main(sys.argv[1:])
