"""Vector drawings (PDF via cairo), 1-bit: pure black on white, no grey.

All coordinates handed to LaTeX are in millimetres from the top-left corner of
the figure, so labels can be typeset by LaTeX in the book's own font.
"""
import math

import cairo
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

MM = 72 / 25.4          # points per millimetre
HAIR = 0.1 / MM         # 0.1 pt, the thinnest line we print, in mm


class Fit:
    """Maps walk coordinates (word units, y up) to figure millimetres (y down)."""

    def __init__(self, xmin, ymin, xmax, ymax, x0, y0, w, h, rotate=False):
        self.rotate = rotate
        if rotate:   # rotate the drawing 90 degrees so its long side is vertical
            xmin, ymin, xmax, ymax = ymin, -xmax, ymax, -xmin
        self.s = min(w / max(xmax - xmin, 1), h / max(ymax - ymin, 1))
        self.ox = x0 + (w - (xmax - xmin) * self.s) / 2 - xmin * self.s
        self.oy = y0 + (h - (ymax - ymin) * self.s) / 2 + ymax * self.s

    def __call__(self, x, y):
        if self.rotate:
            x, y = y, -x
        return self.ox + x * self.s, self.oy - y * self.s


def fit_walk(w, x0, y0, wmm, hmm):
    v = w.vertices()
    xmin, ymin = v.min(0); xmax, ymax = v.max(0)
    rot = (xmax - xmin) > (ymax - ymin) and (hmm > wmm)
    return Fit(xmin, ymin, xmax, ymax, x0, y0, wmm, hmm, rotate=rot)


def _surface(path, wmm, hmm):
    surf = cairo.PDFSurface(path, wmm * MM, hmm * MM)
    ctx = cairo.Context(surf)
    ctx.scale(MM, MM)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_source_rgb(0, 0, 0)
    return surf, ctx


def _polyline(ctx, pts):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)


def _fill_geom(ctx, g):
    if g.is_empty:
        return
    geoms = getattr(g, "geoms", [g])
    for p in geoms:
        if p.geom_type != "Polygon":
            continue
        for ring in [p.exterior] + list(p.interiors):
            c = list(ring.coords)
            ctx.move_to(*c[0])
            for q in c[1:]:
                ctx.line_to(*q)
            ctx.close_path()
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.fill()


def arrow_geometry(x0, y0, x1, y1, shaft=0.18, head=1.4, head_w=0.9):
    """Leader from (x0,y0) to the anchor (x1,y1), with an arrowhead at the anchor (mm)."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    bx, by = x1 - ux * head, y1 - uy * head
    line = LineString([(x0, y0), (bx, by)]).buffer(shaft / 2, cap_style=2)
    tri = Polygon([(x1, y1), (bx - uy * head_w / 2, by + ux * head_w / 2), (bx + uy * head_w / 2, by - ux * head_w / 2)])
    return unary_union([line, tri])


def draw_maze(w, path, wmm, hmm, band=0.0, line_mm=None, anchors=(), labels_out=None,
              label_h=None, chapter_starts=None):
    """The whole book on one page.

    anchors: list of dicts {x, y (walk units), text, lines} - each gets a label in
    the left or right margin band and a leader arrow. The arrow is black, and
    white wherever it lies over the drawing's ink.
    Writes label positions (mm) into labels_out for LaTeX.
    """
    fit = fit_walk(w, band, 0, wmm - 2 * band, hmm)
    lw = line_mm or max(HAIR, 0.12 / MM * 1.0)
    surf, ctx = _surface(path, wmm, hmm)
    v = w.vertices()
    pts = [fit(x, y) for x, y in v]
    ctx.set_line_width(lw)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
    _polyline(ctx, pts)
    ctx.stroke()
    # start and end: the only marks drawn on the path itself
    for (x, y), filled in ((pts[0], True), (pts[-1], False)):
        ctx.rectangle(x - 0.6, y - 0.6, 1.2, 1.2)
        ctx.set_source_rgb(0 if filled else 1, 0 if filled else 1, 0 if filled else 1)
        ctx.fill_preserve(); ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(0.2); ctx.stroke()
    if anchors:
        # annotations never touch the drawing: every arrow points at its place from the
        # margin and stops GAP mm before the first line it would meet
        GAP = 0.8
        ink = unary_union([LineString(pts[k:k + 2]) for k in range(len(pts) - 1)]).buffer(lw / 2 + GAP)
        placed = place_labels([dict(a, px=fit(a["x"], a["y"])[0], py=fit(a["x"], a["y"])[1]) for a in anchors],
                              wmm, hmm, band, label_h or 3.2)
        for a in placed:
            ray = LineString([(a["lx"], a["ly"]), (a["px"], a["py"])])
            hit = ray.intersection(ink)
            ex, ey = a["px"], a["py"]
            if not hit.is_empty:      # first point where the ray would enter the drawing's ink
                coords = [c for g in getattr(hit, "geoms", [hit]) for c in g.coords]
                d = min(ray.project(Point(c)) for c in coords)
                ex, ey = ray.interpolate(d).x, ray.interpolate(d).y
            a["tx"], a["ty"] = ex, ey
            if math.hypot(ex - a["lx"], ey - a["ly"]) > 2.0:
                ctx.set_source_rgb(0, 0, 0)
                _fill_geom(ctx, arrow_geometry(a["lx"], a["ly"], ex, ey).difference(ink))
        if labels_out is not None:
            labels_out.extend(placed)
    surf.finish()
    return fit


def place_labels(anchors, wmm, hmm, band, line_h):
    """Two margin columns. Each label goes to the side its anchor is on, at the
    anchor's height, then labels are pushed apart so they never overlap."""
    cx = wmm / 2
    out = []
    for side in ("L", "R"):
        col = [a for a in anchors if (a["px"] < cx) == (side == "L")]
        col.sort(key=lambda a: a["py"])
        hs = [a.get("lines", 2) * line_h + 1.6 for a in col]
        ys = [a["py"] - hs[k] / 2 for k, a in enumerate(col)]
        for _ in range(200):  # relax: push down, then clamp from the bottom
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + hs[k - 1])
            if ys and ys[-1] + hs[-1] > hmm:
                ys[-1] = hmm - hs[-1]
                for k in range(len(ys) - 2, -1, -1):
                    ys[k] = min(ys[k], ys[k + 1] - hs[k])
            ys = [max(0, y) for y in ys]
        for a, y, h in zip(col, ys, hs):
            lx = band - 1.0 if side == "L" else wmm - band + 1.0
            out.append(dict(a, side=side, top=y, height=h, lx=lx, ly=y + 1.8))
    return out


def draw_zoom(w, path, wmm, hmm, cx, cy, radius, chapter_of, styles, crossings=None, line_mm=0.22):
    """A square window of the walk around (cx, cy), radius in word units.

    styles: chapter index -> dash pattern (mm) or None for solid. Crossing points
    inside the window are marked with small open circles.
    Returns the transform so callers can place labels.
    """
    fit = Fit(cx - radius, cy - radius, cx + radius, cy + radius, 0, 0, wmm, hmm)
    surf, ctx = _surface(path, wmm, hmm)
    ctx.rectangle(0, 0, wmm, hmm); ctx.clip()
    ctx.set_line_width(line_mm)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    win = (cx - radius * 1.5, cy - radius * 1.5, cx + radius * 1.5, cy + radius * 1.5)
    for k in range(w.N):
        x0, y0, x1, y1 = w.x0[k], w.y0[k], w.x1[k], w.y1[k]
        if max(x0, x1) < win[0] or min(x0, x1) > win[2] or max(y0, y1) < win[1] or min(y0, y1) > win[3]:
            continue
        dash = styles.get(chapter_of[k], styles.get("other"))
        ctx.set_dash(dash or [])
        ctx.move_to(*fit(x0, y0)); ctx.line_to(*fit(x1, y1)); ctx.stroke()
    ctx.set_dash([])
    if crossings is not None:
        for x, y, i, j in crossings:
            if abs(x - cx) <= radius and abs(y - cy) <= radius:
                px, py = fit(x, y)
                ctx.arc(px, py, 0.5, 0, 2 * math.pi)
                ctx.set_source_rgb(1, 1, 1); ctx.fill_preserve()
                ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(0.15); ctx.stroke()
    ctx.set_line_width(0.25); ctx.rectangle(0, 0, wmm, hmm); ctx.stroke()
    surf.finish()
    return fit


def draw_mineshaft(v3, path, wmm, hmm, azimuth=35.0, elevation=18.0, dist_factor=2.2, lw_near=0.45, lw_far=0.12,
                   vertical_scale=None, marks=()):
    """The 3D walk from an angled side view, in perspective, as a line drawing.

    Strokes are drawn from far to near; each gets a white casing before its black
    line, so nearer strokes visibly pass in front of farther ones. Line weight
    falls off with distance. vertical_scale compresses depth (as in a geological
    section) so that the shaft's horizontal wandering stays visible; by default
    the depth is scaled to 1.5 x the horizontal extent. Returns the projected
    positions (mm) of the vertices listed in marks, and the scale used.
    """
    v3 = v3.astype(float).copy()
    horiz = max(np.ptp(v3[:, 0]), np.ptp(v3[:, 1]))
    if vertical_scale is None:
        vertical_scale = min(1.0, 1.5 * horiz / max(np.ptp(v3[:, 2]), 1e-9))
    v3[:, 2] *= vertical_scale
    c = (v3.min(0) + v3.max(0)) / 2
    span = np.linalg.norm(v3.max(0) - v3.min(0))
    az, el = math.radians(azimuth), math.radians(elevation)
    eye = c + span * dist_factor * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = c - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rel = v3 - eye
    z = rel @ f
    px, py = (rel @ r) / z, (rel @ u) / z
    xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
    s = min(wmm / (xmax - xmin), hmm / (ymax - ymin)) * 0.98
    X = (px - (xmin + xmax) / 2) * s + wmm / 2
    Y = hmm / 2 - (py - (ymin + ymax) / 2) * s
    surf, ctx = _surface(path, wmm, hmm)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    zm = (z[:-1] + z[1:]) / 2
    order = np.argsort(-zm)
    zn = (zm - zm.min()) / max(zm.max() - zm.min(), 1e-9)   # 0 near .. 1 far
    for k in order:
        lw = lw_near + (lw_far - lw_near) * zn[k]
        x0, y0, x1, y1 = X[k], Y[k], X[k + 1], Y[k + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        if L > 1e-6:   # white casing, shortened at both ends so it never cuts the joints
            e = min(lw * 3.2, 0.3 * L) / L
            ctx.move_to(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e)
            ctx.line_to(x1 - (x1 - x0) * e, y1 - (y1 - y0) * e)
            ctx.set_source_rgb(1, 1, 1); ctx.set_line_width(lw * 3.2); ctx.set_line_cap(cairo.LINE_CAP_BUTT); ctx.stroke()
        ctx.move_to(x0, y0); ctx.line_to(x1, y1)
        ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(max(lw, HAIR)); ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.stroke()
    for k in marks:   # chapter marks: open circles, labelled by LaTeX
        ctx.arc(X[k], Y[k], 0.9, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1); ctx.fill_preserve(); ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.25); ctx.stroke()
    surf.finish()
    return [(float(X[k]), float(Y[k])) for k in marks], vertical_scale


def draw_path(vertices, path, wmm, hmm, line_mm=None, marks=(), fixed=None):
    """Any 2D polyline (float coordinates), fitted to the box, long side vertical.
    fixed=(centre, extent): draw at a common scale instead, extent word units across the box."""
    v = np.asarray(vertices, float)
    if fixed is not None:
        (cx, cy), e = fixed
        fit = Fit(cx - e / 2, cy - e / 2, cx + e / 2, cy + e / 2, 0, 0, wmm, hmm)
    else:
        xmin, ymin = v.min(0); xmax, ymax = v.max(0)
        rot = (xmax - xmin) > (ymax - ymin) and (hmm > wmm)
        fit = Fit(xmin, ymin, xmax, ymax, 0, 0, wmm, hmm, rotate=rot)
    surf, ctx = _surface(path, wmm, hmm)
    ctx.set_line_width(line_mm or max(HAIR, 0.12 / MM))
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    _polyline(ctx, [fit(x, y) for x, y in v])
    ctx.stroke()
    for k, filled in ((0, True), (len(v) - 1, False)):
        x, y = fit(*v[k])
        ctx.rectangle(x - 0.6, y - 0.6, 1.2, 1.2)
        ctx.set_source_rgb(0 if filled else 1, 0 if filled else 1, 0 if filled else 1)
        ctx.fill_preserve(); ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(0.2); ctx.stroke()
    out = []
    for k in marks:
        x, y = fit(*v[k]); out.append((x, y))
        ctx.arc(x, y, 0.55, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1); ctx.fill_preserve(); ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.15); ctx.stroke()
    surf.finish()
    return out, fit.s


def draw_cutout(w, texts, path, wmm, hmm, cx, cy, radius, focus=(), max_font_mm=2.4):
    """A magnified window of the walk around (cx, cy): every stroke is drawn as a thin
    line with its sentence set along it, reading in the walking direction (the text
    stands on the line, on its left-hand side). Strokes in `focus` are heavier and
    set in semibold. radius is the half-width of the window in word units."""
    ry = radius * hmm / wmm
    fit = Fit(cx - radius, cy - ry, cx + radius, cy + ry, 0, 0, wmm, hmm)
    s = fit.s                                # mm per word
    fs = min(max_font_mm, 0.62 * s)          # type size follows the scale
    surf, ctx = _surface(path, wmm, hmm)
    ctx.rectangle(0, 0, wmm, hmm); ctx.clip()
    focus = set(focus)
    win = (cx - radius - 60, cy - ry - 60, cx + radius + 60, cy + ry + 60)
    order = [k for k in range(w.N) if k not in focus] + sorted(focus)
    for k in order:
        x0, y0, x1, y1 = w.x0[k], w.y0[k], w.x1[k], w.y1[k]
        if max(x0, x1) < win[0] or min(x0, x1) > win[2] or max(y0, y1) < win[1] or min(y0, y1) > win[3]:
            continue
        a, b = fit(x0, y0), fit(x1, y1)
        heavy = k in focus
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.32 if heavy else 0.1)
        ctx.move_to(*a); ctx.line_to(*b); ctx.stroke()
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1 or fs < 0.9:
            continue
        ctx.select_font_face("EB Garamond", cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD if heavy else cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(fs)
        ext = ctx.text_extents(texts[k])
        if ext.x_advance <= 0:
            continue
        ctx.save()
        ctx.translate(*a)
        ctx.rotate(math.atan2(b[1] - a[1], b[0] - a[0]))
        ctx.scale(L / ext.x_advance, 1.0)    # the sentence fills its stroke exactly
        ctx.move_to(0, -0.25)
        ctx.show_text(texts[k])
        ctx.restore()
    ctx.reset_clip()
    ctx.set_line_width(0.25); ctx.rectangle(0, 0, wmm, hmm); ctx.stroke()
    surf.finish()
    return fit
