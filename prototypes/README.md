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
| `view3d.py texts/7849.txt out.png [n] [sun_az] [sun_el] [haze]` | Ground view inside a pathway, with wall heights set by the words, sunlit through haze, 1-bit. n picks the n-th ranked viewpoint; sun azimuth is relative to the view direction → `ground_view_trial.png`, `ground_view_1bit_detail.png` (actual 1-bit pixels) |

Also needed for `view3d.py`: `fonts/EBGaramond.ttf` (Google Fonts, SIL Open Font License), and `pip install numpy pillow numba`.
