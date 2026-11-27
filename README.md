# make typographic art from books


<a href="https://www.buymeacoffee.com/sloev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-pink.png" alt="Buy Me A Coffee" height="51px" width="217px"></a>


uses sentiment of words to make turns

## dependencies

```
python 3+

brew reinstall cairo pkg-config

pip install cairocffi afinn

npm install svgexport -g

```

## usage

```
# get a text file with lots of text inside and run

python main.py input.txt

# now you have a huge output.svg
# create a portable downscaled PNG with correct ratio and max 1000 in width

svgexport output.svg output2.png 1000:
```

## here is Kafkas *The Trial* as a typographic maze:

![kafkas The Trial](/the_trial.png?raw=true "Kafkas The Trial")

And here is an excerpt showing that the lines actually are text:

![kafkas The Trial](/the_trial_excerpt.svg?raw=true&sanitize=true "Kafkas The Trial excerpt")


## The engine (labyrinthography)

`labyrinth/` turns a Project Gutenberg text into the analysis and figures for the book, and `book/` typesets them with LuaLaTeX on A5.

```
pip install numpy shapely pycairo pyyaml numba pillow wordfreq
apt-get install wamerican   # the dictionary for the vocabulary record
make            # = python3 -m labyrinth.build + labyrinth.experiments for each books/*.yaml, then latexmk in book/
```

| Module | What it does |
|---|---|
| `labyrinth/text.py` | Fetches and cleans the text, detects chapters and subtitles, splits sentences (initials such as "K." and honorifics don't end a sentence) |
| `labyrinth/sentiment.py` | AFINN-165 sentence score → turn (left / straight / right) |
| `labyrinth/walk.py` | The 2D walk (exact integer geometry) and the 3D mineshaft (left turns tilt up, right turns tilt down) |
| `labyrinth/analysis.py` | Global measures, the shuffle baseline (z-scores, empirical p), crossings, palimpsests, knots, hub, return, coil, darkest passage, corridor, chapter overlap |
| `labyrinth/draw.py` | Vector PDFs, pure black: the labelled maze (arrows white where they cross ink), knot zooms, the mineshaft |
| `labyrinth/build.py` | Runs everything for one `books/*.yaml`, writes `build/<slug>/` including `numbers.tex`, so every number in the prose is generated |
| `labyrinth/variants.py` | Turning-rule variations: quarter (90°), graded (VADER-normalised sentence valence × 90°), rolling (per word, running mean of the last 12 words) |
| `labyrinth/letters.py` | Standing letters for any path geometry, screened to 1-bit |
| `labyrinth/experiments.py` | Every rule from every view (plan, mineshaft, letters at eye level and from above) → `build/<slug>/variants/` |
| `labyrinth/vocab.py` | The vocabulary record: the dictionary (system word list, ranked by `wordfreq` commonness) as a spiral disc, common words at the centre; a book's words are black pixels |
| `labyrinth/instruments.py` | Tree rings (chapters as rings, word length as grain), the loom (the cast as woven cloth), echoes (each word's longest absence as an arch), the tide (speech inside quotation marks) |
| `labyrinth/instruments.py` (more) | Breath (clause lengths), the punctuation sky (how sentences end), the pronoun walk (I north, you south, he east, she west), the clock (times of day on a 24-hour dial), the light meter (light against dark words, screened to 1-bit) |
| `labyrinth/compare.py` | All books side by side: drawings, records, rings, looms, echoes, tides and the numbers → `build/compare/` |
| `book/` | `main.tex`, `methods.tex` (definitions and the test), `trial.tex` (The Trial), `labyrinth.sty` (A5 page design) |

`book/main.pdf` is the current specimen.

## The zine

`zine/` is a 40-page A5 zine version of the book: dense grids of drawings for all five books, black only.

```
make && make zine
```

- `zine/zine.pdf`: reading order, for screen and proofs.
- `zine/zine-print.pdf`: 10 A4 sheets, imposed for saddle stitching. Print double-sided, flip on the short edge,
  stack in order, fold, staple twice on the fold.

The Trial pages quote and draw David Wyllie's translation (a copyrighted Project Gutenberg eBook);
get permission before selling copies.
