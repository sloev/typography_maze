# Prototypes

Quick experiments behind PLAN.md. Not the final engine.

Expected inputs (not committed):
- `texts/<gutenberg-id>.txt`: Project Gutenberg plain-text files, e.g. 7849 (The Trial), 5200 (Metamorphosis), 1952 (The Yellow Wallpaper)
- `texts/AFINN-en-165.txt`: the AFINN lexicon (https://github.com/fnielsen/afinn)

Run them from the folder that contains `texts/`.

| Script | Output |
|---|---|
| `strokes.py texts/*.txt` | One stroke per sentence, fitted to A5 → `sentence_strokes_a5.png` |
| `crossings.py texts/*.txt` | Crossings, crossing gaps, collinear overlaps, most-crossed sentence |
| `view3d.py texts/7849.txt out.png [n] [sun_az] [sun_el] [haze]` | Ground view inside a pathway, with wall heights set by the words, sunlit through haze, 1-bit. n picks the n-th ranked viewpoint (ranked by crossings ahead); sun azimuth is relative to the view direction → `ground_view_trial.png`, `ground_view_1bit_detail.png` (actual 1-bit pixels) |

Also needed for `view3d.py`: `fonts/EBGaramond.ttf` (Google Fonts, SIL Open Font License), and `pip install numpy pillow numba`.

### Viewpoints

`render.py texts/7849.txt out.png [n] [sun_az] [sun_el] [haze] [mode]`, where mode is one of
`walk corner worm wall oblique aerial plan`. It uses `scene.py` (walk, crossings, word-height field, lettering)
and renders any camera position and tilt by per-pixel ray casting → `viewpoints_trial.png`
(top to bottom: wall, oblique / aerial, plan / worm, corner).

### Standing letters (current 3D concept)

`letters3d.py texts/7849.txt out.png [n] [sun_az] [sun_el] [mode]`, where mode is one of `stand among high low`.
Sentences stand on the ground as black letters with no thickness, along their strokes; there are no walls.
Output is screened to 1-bit with `tools/halftone.py` (the grudgebook method: stochastic clustered dots, 1200 dpi, A5 text width).
Set `SCREEN=fm` for randomised error diffusion instead.
n picks the n-th busiest junction → `standing_letters.png`, plus `standing_letters_1200dpi_detail.png` (a 1:1 crop of the 1200 dpi bitmap)
(rows: stand / among, high / low, Metamorphosis / Yellow Wallpaper).
