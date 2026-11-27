"""Standing letters: every stroke's text stands on the ground along the stroke as flat
black cut-out letters (no thickness, no walls), lit by sky and sun, screened to 1-bit.

Works for any stroke headings (quarter, graded and rolling rules). Stroke lengths
must be in set-width units (the text's natural width at em = EM), so letters keep
their proportions.
"""
import math
import os
import sys

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

from .text import CACHE

EM = 1.5
PPU = 110
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def font():
    f = ImageFont.truetype(os.path.join(CACHE, "fonts", "EBGaramond-SemiBold.otf"), int(EM * PPU))
    return f


def set_widths(texts, add_space=False):
    f = font()
    return np.array([f.getlength(t + (" " if add_space else "")) / PPU for t in texts])


def _segdist(P, px, py):
    ax, ay, bx, by = P.T
    vx, vy = bx - ax, by - ay
    tt = np.clip(((px - ax) * vx + (py - ay) * vy) / np.maximum(vx * vx + vy * vy, 1e-9), 0, 1)
    return np.hypot(ax + tt * vx - px, ay + tt * vy - py)


def busiest_point(path, radius=8.0, candidates=1500, seed=3):
    """The point where the most distinct sentences pass within `radius`."""
    v = path.vertices2d()
    P = np.hstack([v[:-1], v[1:]])
    mids = (v[:-1] + v[1:]) / 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mids), size=min(candidates, len(mids)), replace=False)
    best = (-1, 0)
    for k in idx:
        near = _segdist(P, *mids[k]) < radius
        n = len(np.unique(path.group[near]))
        if n > best[0]:
            best = (n, k)
    return best[0], int(best[1]), mids[best[1]]


@njit(cache=True)
def _hit(ox, oy, oz, dx, dy, dz, B, ATL, PPU, ASC, HP, tmax, anyhit):
    best = tmax; kind = 0
    if dz < 0:
        tg = -oz / dz
        if tg < best:
            best = tg; kind = 1
    for b in range(B.shape[0]):
        x0 = B[b, 0]; y0 = B[b, 1]; ux = B[b, 2]; uy = B[b, 3]; L = B[b, 4]; off = int(B[b, 5])
        lx = -uy; ly = ux
        dp = dx * lx + dy * ly
        if abs(dp) < 1e-12:
            continue
        rp = (ox - x0) * lx + (oy - y0) * ly
        t = -rp / dp
        if t <= 1e-6 or t >= best:
            continue
        s_ = (ox - x0) * ux + (oy - y0) * uy + (dx * ux + dy * uy) * t
        z_ = oz + dz * t
        if s_ < 0 or s_ >= L or z_ < 0 or z_ > ASC:
            continue
        fx = s_ * PPU - 0.5; fz = (ASC - z_) * PPU - 0.5
        ix = int(math.floor(fx)); iz = int(math.floor(fz)); ax = fx - ix; az = fz - iz
        Lp = int(L * PPU)
        if iz < 0 or iz + 1 >= HP or ix < 0 or ix + 1 >= Lp:
            continue
        cov = (ATL[iz, off + ix] * (1 - ax) * (1 - az) + ATL[iz, off + ix + 1] * ax * (1 - az)
               + ATL[iz + 1, off + ix] * (1 - ax) * az + ATL[iz + 1, off + ix + 1] * ax * az)
        if cov >= 127.5:
            if anyhit:
                return t, 2
            best = t; kind = 2
    return best, kind


@njit(cache=True)
def _render(cx, cy, cz, fwd, right, up, f, W, H, B, ATL, PPU, ASC, HP, sx, sy, sz, DEP, KND, LIT, NDS):
    for r in range(H):
        for c in range(W):
            u = (c - W / 2 + 0.5) / f; v = (H / 2 - r - 0.5) / f
            dx = fwd[0] + u * right[0] + v * up[0]; dy = fwd[1] + u * right[1] + v * up[1]
            dz = fwd[2] + u * right[2] + v * up[2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= n; dy /= n; dz /= n
            t, k = _hit(cx, cy, cz, dx, dy, dz, B, ATL, PPU, ASC, HP, 400.0, False)
            KND[r, c] = k; DEP[r, c] = t; NDS[r, c] = dx * sx + dy * sy + dz * sz
            if k != 1:
                continue
            px = cx + dx * t; py = cy + dy * t
            t2, k2 = _hit(px, py, 0.03, sx, sy, sz, B, ATL, PPU, ASC, HP, 80.0, True)
            LIT[r, c] = sz if k2 == 0 else 0.0


def render(path, out_png, view="stand", sun_az=150.0, sun_el=35.0, W=2000, H=1333, width_in=None,
           radius=8.0, scene_radius=60.0, haze=55.0):
    """path: lengths in set-width units. Writes out_png (1-bit, screened) and *_tone.png."""
    v = path.vertices2d()
    n_sent, k, (jx, jy) = busiest_point(path, radius)
    near = np.nonzero(_segdist(np.hstack([v[:-1], v[1:]]), jx, jy) < scene_radius)[0]
    f_ = font()
    asc, desc = f_.getmetrics()
    ASC = asc / PPU
    HP = asc + desc + 2
    offs, total = [], 0
    for i in near:
        offs.append(total); total += int(path.length[i] * PPU) + 4
    atlas = np.zeros((HP, max(total, 1)), np.uint8)
    for i, o in zip(near, offs):
        w = int(path.length[i] * PPU) + 4
        im = Image.new("L", (w, HP), 0)
        ImageDraw.Draw(im).text((0, 0), path.text[i], font=f_, fill=255)
        atlas[:, o:o + w] = np.asarray(im)
    h = np.radians(path.heading[near])
    B = np.stack([v[near, 0], v[near, 1], np.cos(h), np.sin(h), path.length[near], np.array(offs, float)], 1)
    # local direction at the junction: the stroke passing closest to it
    d = _segdist(np.hstack([v[:-1], v[1:]]), jx, jy)
    base = int(np.argmin(d))
    bx, by = math.cos(math.radians(path.heading[base])), math.sin(math.radians(path.heading[base]))
    lx, ly = -by, bx
    if view == "stand":    # on the readable (right-hand) side, looking along the stroke into the junction
        cam = (jx - bx * 7 - lx * 3.2, jy - by * 7 - ly * 3.2, 0.62); tgt = (jx + bx * 2, jy + by * 2, 0.6); fv = 66
    elif view == "high":   # above and behind, looking down on the junction
        cam = (jx - bx * 11 - lx * 9, jy - by * 11 - ly * 9, 9.0); tgt = (jx, jy, 0.0); fv = 58
    else:
        raise ValueError(view)
    cx, cy, cz = cam
    fwd = np.array([tgt[0] - cx, tgt[1] - cy, tgt[2] - cz]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    f = (W / 2) / math.tan(math.radians(fv) / 2)
    az = math.atan2(fwd[1], fwd[0]) + math.radians(sun_az); el = math.radians(sun_el)
    sun = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    DEP, LIT, NDS = [np.zeros((H, W)) for _ in range(3)]
    KND = np.zeros((H, W), np.int8)
    _render(cx, cy, cz, fwd, right, up, f, W, H, B, atlas, float(PPU), ASC, HP, sun[0], sun[1], sun[2],
            DEP, KND, LIT, NDS)
    col = np.where(KND == 1, 0.98 * (0.55 + 0.62 * LIT / max(sun[2], 1e-6) * sun[2]), 0.0)
    col = np.where(KND == 0, 0.92 + 0.08 * np.clip(NDS, 0, 1) ** 8, col)
    DH = np.where(KND == 2, np.maximum(DEP - 25.0, 0.0), DEP)
    Tr = np.exp(-np.where(KND == 0, 1e9, DH) / haze)
    img = np.clip(col * Tr + 0.96 * (1 - Tr), 0, 1)
    tone = out_png.replace(".png", "_tone.png")
    Image.fromarray((img * 255).astype(np.uint8)).save(tone)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import halftone
    halftone.screen_file(tone, out_png, width_in=width_in or halftone.A5_TEXT_WIDTH_IN)
    return dict(junction=(float(jx), float(jy)), sentences_near=int(n_sent), strokes_in_scene=int(len(near)),
                base_group=int(path.group[base]))
