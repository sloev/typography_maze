#!/usr/bin/env python3
"""1-bit screening for the book's raster images (the ground views).

The same method as the sister project's tools/halftone.py: every raster
image is printed as a 1-bit bitmap at its exact placed size, at 1200 dpi, so
the press never re-screens it.

Pipeline: levels (0.5 / 99.5 percentile) → resample to the device grid →
unsharp mask → gentle S-curve → ink coverage → inverse dot gain (12 %) →
clean paper white / solid black → screen.

Screens (tools/halftone.c, compiled on first use):
  dots  stochastic clustered dots (default): dot centres thrown at random with
        a minimum spacing, dots growing from each centre as the tone darkens;
        no grid, rows or lattice. Mean spacing 220 dots per inch.
  fm    randomised serpentine Floyd–Steinberg: threshold and weights jittered
        per pixel, so flat tones show no lattice or worms.

Usage:
    python3 tools/halftone.py in.png out.png --width-in 5.04 [--screen dots|fm]
or from Python: screen_array(gray01, width_in) -> bool ink array.
"""
import argparse
import ctypes
import os
import subprocess
import tempfile
import zlib

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
A5_TEXT_WIDTH_IN = (148 - 2 * 10) / 25.4   # A5 with 10 mm side margins


def _lib():
    src = os.path.join(HERE, "halftone.c")
    so = os.path.join(tempfile.gettempdir(), "typomaze-halftone-%d.so" % int(os.path.getmtime(src)))
    if not os.path.exists(so):
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", so, src, "-lm"], check=True)
    lib = ctypes.CDLL(so)
    lib.diffuse.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                            ctypes.c_double, ctypes.c_double, ctypes.c_uint64]
    lib.dot_field.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                              ctypes.c_double, ctypes.c_double, ctypes.c_uint64]
    return lib


def dot_threshold(h, w, seed, spacing, rmin=0.62, wjit=0.35):
    f = np.empty((h, w), dtype=np.float32)
    _lib().dot_field(f.ctypes.data, h, w, spacing, rmin, wjit, seed)
    hist, edges = np.histogram(f, bins=8192)
    cdf = np.concatenate([[0.0], np.cumsum(hist) / f.size])
    return np.interp(f, edges, cdf).astype(np.float32)


def diffuse(cov, seed, tjit=0.25, wjit=0.5):
    buf = np.ascontiguousarray(cov, dtype=np.float64).copy()
    out = np.zeros(buf.shape, dtype=np.uint8)
    _lib().diffuse(buf.ctypes.data, out.ctypes.data, buf.shape[0], buf.shape[1], tjit, wjit, seed)
    return out.astype(bool)


def dot_gain_inverse(p, g):
    if g <= 0:
        return p
    a = 1 + 4 * g
    return (a - np.sqrt(a * a - 16 * g * p)) / (8 * g)


def screen_array(gray, width_in=A5_TEXT_WIDTH_IN, dpi=1200, gain=0.12, kind="dots",
                 spacing_dpi=220, seed=1, resample=Image.LANCZOS):
    """gray: 2D array, 0 = black .. 1 = white. Returns a bool array, True = ink."""
    h0, w0 = gray.shape
    w = round(width_in * dpi); h = round(w * h0 / w0)
    a = np.asarray(gray, dtype=float)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8), "L").resize((w, h), resample)
    im = im.filter(ImageFilter.UnsharpMask(radius=dpi / 400, percent=130, threshold=2))
    a = np.asarray(im, dtype=float) / 255
    a = a + 0.3 * (a * a * (3 - 2 * a) - a)
    cov = 1 - np.clip(a, 0, 1)
    cov = dot_gain_inverse(cov, gain)
    cov[cov < 0.012] = 0
    cov[cov > 0.988] = 1
    if kind == "fm":
        return diffuse(cov, seed | 1)
    return cov > dot_threshold(h, w, seed | 1, dpi / spacing_dpi)


def screen_file(src, dst, **kw):
    gray = np.asarray(Image.open(src).convert("L"), dtype=float) / 255
    kw.setdefault("seed", zlib.crc32(os.path.basename(src).encode()))
    ink = screen_array(gray, **kw)
    dpi = kw.get("dpi", 1200)
    Image.fromarray(~ink).convert("1").save(dst, dpi=(dpi, dpi), optimize=True)
    return ink.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--width-in", type=float, default=A5_TEXT_WIDTH_IN)
    ap.add_argument("--dpi", type=int, default=1200)
    ap.add_argument("--gain", type=float, default=0.12)
    ap.add_argument("--screen", choices=["dots", "fm"], default="dots")
    ap.add_argument("--spacing", type=float, default=220)
    a = ap.parse_args()
    h, w = screen_file(a.src, a.dst, width_in=a.width_in, dpi=a.dpi, gain=a.gain,
                       kind=a.screen, spacing_dpi=a.spacing)
    print(f"{a.dst}  {w}x{h} px at {a.dpi} dpi")
