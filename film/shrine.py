"""Roadside earth-god shrine (tudi miao) of the Guilin / Guangxi countryside: a real-scale foreground prop.

Reference (the small brick shrines at village entrances, field edges and road forks): a grey fired-brick
body (1.6 m wide, 1.2 m deep) on a granite-and-cement plinth; the side walls run forward into a shallow
porch and end in stepped brick corbels (chitou) under the eaves; a hard-gable roof of small grey clay tiles
laid yin-yang (concave pan rows, convex cover rows), lime-plastered gable copings and a plastered ridge whose
ends sweep up into curls (Lingnan "dragon-boat" ridge), a little gourd on the ridge.  The doorway (granite
frame) opens into a dark sooty niche with a painted seated Tudi Gong on a red pedestal and a red paper
spirit tablet.  Red paper couplets on both sides of the door with last year's faded remnants underneath,
a horizontal banner above, a red cloth swag over the lintel; on the stone altar block in the porch a bronze
burner full of ash and red incense stubs with three lit sticks, two red candle stubs, mandarins and a stack
of joss paper; cups of rice wine on the sill; a cracked slab apron in front with firecracker shells, burnt
red paper and a black ash patch where paper money was burnt; weeds round the base.

Weathering is in the shaders (object space, so every instance keeps it): damp dark base with moss, rain
streaks running down from the copings, black soot above the doorway from decades of incense, algae in the
pan-tile channels near the eaves, black mould on the lime plaster, faded and torn paper.

Everything is one mesh built with the village `MB` builder (plus per-face attributes for the shaders and UVs
for the paper); about 13k faces, 8 materials.
"""
import math
import os
from contextlib import contextmanager

import bpy
import numpy as np

from .village import MB, Wall

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# ------------------------------------------------------------------ layout (local: x across the front, front faces -y,
# z up; origin = front-centre of the paved apron at ground level, the shrine body stands 0.85-1.95 m behind it)
HW = 0.80                    # half width of the body (outer faces of the side walls)
XI = 0.60                    # inner faces of the side walls
Y_PORCH = 0.85               # front ends of the side walls (porch mouth)
Y_FRONT = 1.10               # outer face of the doorway wall (porch 25 cm deep)
T_FRONT = 0.15
Y_BACK = 1.95
T_BACK = 0.18
Z_PL = 0.22                  # plinth top = porch floor
Z_SILL, Z_LINTEL, DOOR_HW = 0.50, 1.08, 0.23
Z_ALTAR = 0.47               # top of the stone altar slab in the porch
E = 1.45                     # underside of the roof deck at the front eave line
PITCH = 28.0
DECK_T = 0.03
Y_R = 0.5 * (Y_PORCH + Y_BACK)          # ridge line
Y_EF = Y_PORCH - 0.15                   # front eave (deck edge)
Y_EB = Y_BACK + 0.12                    # back eave
TAN = math.tan(math.radians(PITCH))
Z_R = E + DECK_T + TAN * (Y_R - Y_EF)   # deck top under the ridge
APRON_Y = Y_PORCH - 0.08                # apron runs from y=0 to the plinth front

# roof tiles (small grey clay tiles, real size)
T_T = 0.011                  # tile thickness
PAN_W, SAG = 0.18, 0.026     # concave pan tiles (channels)
COV_W, RISE = 0.13, 0.034    # convex cover tiles over the joints
STEP = 0.062                 # exposed length per tile course (lay "seven hidden, three showing")
ROW = 0.20                   # pan row pitch


# ------------------------------------------------------------------ mesh builder with per-face attributes and UVs
class SB(MB):
    """MB plus per-face shader attributes (grime, tvar, nch, col, rgh, met, emit), per-loop UVs, smooth flags
    and shared-vertex grids (for smooth curved surfaces)."""

    KEYS = ("grime", "tvar", "nch", "col", "rgh", "met", "emit", "smooth")

    def __init__(self):
        super().__init__()
        self.uv = []
        self.fd = []
        self.s = dict(grime=0.0, tvar=0.0, nch=0.0, col=(0.5, 0.5, 0.5), rgh=0.6, met=0.0, emit=0.0, smooth=False)

    @contextmanager
    def st(self, **kw):
        old = dict(self.s)
        self.s.update(kw)
        try:
            yield
        finally:
            self.s = old

    def _face(self, n, uv):
        self.fd.append(tuple(self.s[k] for k in self.KEYS))
        self.uv.extend(uv if uv is not None else [(0.0, 0.0)] * n)

    def poly(self, pts, mat, uv=None):
        super().poly(pts, mat)
        self._face(len(pts), uv)

    def grid(self, P, mat, uv=None, closed=False):
        """Quads between rows of a (rows, cols, 3) point array; vertices are shared (smooth shading works)."""
        P = np.asarray(P, float)
        nr, nc = P.shape[:2]
        i0 = len(self.v)
        self.v.extend(map(tuple, P.reshape(-1, 3)))
        for r in range(nr - 1):
            for c in range(nc if closed else nc - 1):
                c2 = (c + 1) % nc
                self.f.append([i0 + r * nc + c, i0 + r * nc + c2, i0 + (r + 1) * nc + c2, i0 + (r + 1) * nc + c])
                self.m.append(mat)
                self._face(4, None if uv is None else [tuple(uv[r][c]), tuple(uv[r][c2]), tuple(uv[r + 1][c2]), tuple(uv[r + 1][c])])

    def lathe(self, prof, c, mat, segs=12, cap_bot=False, cap_top=False, sz=1.0):
        """Surface of revolution round a vertical axis at c: prof = [(r, z), ...] from bottom to top."""
        a = np.linspace(0, 2 * math.pi, segs, endpoint=False)
        P = np.array([[(c[0] + r * math.cos(t), c[1] + r * math.sin(t), c[2] + z * sz) for t in a] for (r, z) in prof])
        self.grid(P, mat, closed=True)
        if cap_bot:
            self.poly(list(P[0][::-1]), mat)
        if cap_top:
            self.poly(list(P[-1]), mat)

    def sweep(self, path, prof, mat, side=(0, 1, 0), scale=None, caps=True):
        """Closed profile [(a, b), ...] (a along `side`, b 'up') swept along a 3D path."""
        path = np.asarray(path, float)
        n = len(path)
        scale = np.ones((n, 2)) if scale is None else np.asarray(scale, float)
        fr = []
        for i in range(n):                          # frames continuous along the path, 'up' at mid-path
            T = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
            T /= np.linalg.norm(T)
            A = np.asarray(side, float) - T * np.dot(side, T)
            A /= np.linalg.norm(A)
            B = np.cross(T, A)
            if fr and np.dot(B, fr[-1][1]) < 0:
                B = -B
            fr.append((A, B))
        flip = -1.0 if fr[n // 2][1][2] < 0 else 1.0
        P = np.array([[path[i] + a * scale[i, 0] * fr[i][0] + flip * b * scale[i, 1] * fr[i][1] for (a, b) in prof]
                      for i in range(n)])
        self.grid(P, mat, closed=True)
        if caps:
            self.poly(list(P[0][::-1]), mat)
            self.poly(list(P[-1]), mat)

    def prism_x(self, outline, x0, x1, mat):
        """Polygon outline [(y, z), ...] (counter-clockwise seen from +x) extruded from x0 to x1."""
        a = [(x1, y, z) for (y, z) in outline]
        b = [(x0, y, z) for (y, z) in outline]
        self.poly(a, mat)
        self.poly(b[::-1], mat)
        k = len(outline)
        for i in range(k):
            j = (i + 1) % k
            self.poly([b[i], b[j], a[j], a[i]], mat)

    def make(self, name, mats, recalc=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        for mt in mats:
            me.materials.append(mt)
        me.polygons.foreach_set("material_index", np.asarray(self.m, dtype=np.int32))
        fd = self.fd
        for k, i in (("grime", 0), ("tvar", 1), ("nch", 2), ("rgh", 4), ("met", 5), ("emit", 6)):
            at = me.attributes.new(k, "FLOAT", "FACE")
            at.data.foreach_set("value", np.array([f[i] for f in fd], np.float32))
        at = me.attributes.new("col", "FLOAT_COLOR", "FACE")
        at.data.foreach_set("color", np.array([(*f[3], 1.0) for f in fd], np.float32).ravel())
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(self.uv, np.float32).ravel())
        me.polygons.foreach_set("use_smooth", np.array([f[7] for f in fd], bool))
        me.validate(clean_customdata=False)
        me.update()
        return me


# ------------------------------------------------------------------ roof surface
def _D(s):
    return (Y_R - Y_EF) if s < 0 else (Y_EB - Y_R)


def zdeck(y):
    """Top of the roof deck: slightly concave slopes (steeper at the ridge, flatter at the eaves)."""
    s = -1 if y < Y_R else 1
    D = _D(s)
    d = abs(y - Y_R)
    a = TAN * 8 / 7
    b = a / (8 * D)
    return Z_R - a * d + b * d * d


def _deck_n(y):
    s = -1 if y < Y_R else 1
    D = _D(s)
    d = abs(y - Y_R)
    a = TAN * 8 / 7
    g = a - 2 * (a / (8 * D)) * d
    n = np.array([0.0, s * g, 1.0])
    return n / np.linalg.norm(n)


def _RP(x, y, h):
    """Point at height h above the deck top (along the deck normal)."""
    n = _deck_n(y)
    return (x, y + n[1] * h, zdeck(y) + n[2] * h)


def _sag(x, y):
    """An old roof sags a little between the gables, most in the middle of each slope."""
    s = -1 if y < Y_R else 1
    t = min(1.0, abs(y - Y_R) / _D(s))
    return -0.011 * math.cos(math.pi * x / 1.5) * math.sin(math.pi * t)


def _pan_h(u):
    return T_T + SAG * (2 * u / PAN_W) ** 2


_COV_BASE = T_T + SAG * (2 * (ROW / 2 - COV_W / 2) / PAN_W) ** 2      # pan surface under the cover edges


def _cov_h(u):
    return _COV_BASE + T_T + RISE * max(0.0, 1 - (2 * u / COV_W) ** 2) ** 0.7


# ------------------------------------------------------------------ parts
def _roof(sb, M, rng):
    us_p = np.linspace(-PAN_W / 2, PAN_W / 2, 7)
    us_c = np.linspace(-COV_W / 2, COV_W / 2, 7)
    for s in (-1, 1):                                                     # front slope, back slope
        D = _D(s)
        y_eave = Y_R + s * D
        # deck (plaster underside, sooty), sampled so it follows the concave slope
        ys = np.linspace(y_eave, Y_R, 9)
        top = [[(x, y, zdeck(y)) for x in (-HW, HW)] for y in ys]
        bot = [[(x, y, zdeck(y) - DECK_T) for x in (-HW, HW)] for y in ys]
        with sb.st(grime=0.55):
            sb.grid(bot, M["plaster"])
            sb.grid(top, M["plaster"])
            sb.poly([bot[0][0], bot[0][1], top[0][1], top[0][0]], M["plaster"])
        # tile rows
        rows = [("pan", x) for x in np.arange(-0.60, 0.601, ROW)] + [("cov", x) for x in np.arange(-0.50, 0.501, ROW)]
        for kind, xc in rows:
            us = us_p if kind == "pan" else us_c
            hf = _pan_h if kind == "pan" else _cov_h
            n_t = int(math.ceil((D + 0.035 - 0.05) / STEP))
            yk = [y_eave + s * 0.035 - s * STEP * k + (rng.normal(0, 0.004) if 0 < k < n_t else 0.0) for k in range(n_t + 1)]
            yk[-1] = Y_R
            xr = xc + rng.normal(0, 0.003)
            wob = 0.0
            for k in range(n_t):
                ya, yb = yk[k], yk[k + 1]
                wob = 0.75 * wob + rng.normal(0, 0.0035)                  # rows wander a little
                xr = xc + wob
                tv = rng.random()
                if rng.random() < 0.03:
                    tv = 1.5                                              # odd newer replacement tile (lighter)
                skew = rng.normal(0, 0.003) + (rng.normal(0, 0.012) if rng.random() < 0.05 else 0.0)
                xa, xb = xr + rng.normal(0, 0.002), xr + skew
                lift = rng.normal(0, 0.0012)
                g = (0.75 if kind == "pan" else 0.25) * (1.0 - 0.6 * k / n_t)
                ha, hb = _sag(xa, ya), _sag(xb, yb)
                with sb.st(tvar=tv, grime=g, smooth=True):
                    sb.grid([[_RP(xa + u, ya, hf(u) + T_T * 0.9 + lift + ha) for u in us],
                             [_RP(xb + u, yb, hf(u) + hb) for u in us]], M["tile"])
                with sb.st(tvar=tv, grime=g):
                    drop = T_T if k == 0 else 0.0
                    sb.grid([[_RP(xa + u, ya, hf(u) + T_T * 0.9 + lift + ha) for u in us],
                             [_RP(xa + u, ya, hf(u) - drop + ha) for u in us]], M["tile"])
                    if k == 0:                                            # underside of the overhanging eave tile
                        sb.grid([[_RP(xa + u, ya, hf(u) - T_T) for u in us],
                                 [_RP(xa + u, y_eave, hf(u) - T_T) for u in us]], M["tile"])
                if k == 0 and kind == "cov":                              # lime mortar plug in the cover-row end
                    with sb.st(grime=0.3 + 0.3 * rng.random()):
                        arc = [_RP(xa + u, ya - s * 0.004, hf(u) + T_T * 0.6) for u in us]
                        base = _RP(xa, ya - s * 0.004, _COV_BASE)
                        sb.poly(arc + [base], M["plaster"])
        # rafters under the overhangs and the porch
        y_in = Y_FRONT if s < 0 else Y_BACK
        with sb.st(col=(0.06, 0.045, 0.035), rgh=0.85):
            for xr_ in np.arange(-0.50, 0.501, 0.20):
                ya, yb = y_eave - s * 0.01, y_in
                za, zb = zdeck(ya) - DECK_T, zdeck(yb) - DECK_T
                q = [(xr_ - 0.02, ya, za - 0.045), (xr_ + 0.02, ya, za - 0.045), (xr_ + 0.02, yb, zb - 0.045), (xr_ - 0.02, yb, zb - 0.045),
                     (xr_ - 0.02, ya, za), (xr_ + 0.02, ya, za), (xr_ + 0.02, yb, zb), (xr_ - 0.02, yb, zb)]
                sb.hexa(q, M["smalls"])


def _ridge(sb, M, rng):
    """Vernacular ridge: mortar bed, a row of grey tiles set on edge leaning towards the middle, a thin mortar
    cap, plastered ridge ends sweeping up into curls over the gables, a gourd on a little pedestal mid-ridge."""
    pl = M["plaster"]
    zb = Z_R + 0.06                                                        # top of the bed
    bed = [(-0.16, -0.05), (0.16, -0.05), (0.10, 0.025), (0.068, 0.06), (-0.068, 0.06), (-0.10, 0.025)]
    with sb.st(grime=0.6):
        sb.sweep([(-HW + 0.03, Y_R, Z_R), (HW - 0.03, Y_R, Z_R)], bed, pl)
    # tiles on edge, leaning 28 deg towards the middle, between the ornament and the ridge ends
    XT, H, lean = 0.53, 0.095, math.radians(28)
    for sx in (-1, 1):
        for x in np.arange(0.085, XT, 0.026):
            xc = sx * (x + rng.normal(0, 0.002))
            ph = lean + rng.normal(0, 0.05)
            up = np.array([-sx * math.sin(ph), 0.0, math.cos(ph)])
            th = np.array([math.cos(ph), 0.0, sx * math.sin(ph)]) * 0.006
            yw = 0.052 + rng.normal(0, 0.002)
            b0 = np.array([xc, Y_R, zb - 0.01])
            c = []
            for hh in (0.0, H):
                for (dt, dy) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    c.append(b0 + up * hh + th * dt + np.array([0.0, dy * yw, 0.0]))
            with sb.st(tvar=rng.random(), grime=0.35):
                sb.hexa(c, M["tile"])
    zc = zb - 0.01 + H * math.cos(lean)
    cap = [(-0.062, -0.006), (0.062, -0.006), (0.062, 0.012), (0.045, 0.024), (-0.045, 0.024), (-0.062, 0.012)]
    with sb.st(grime=0.65):
        sb.sweep([(-XT - 0.02, Y_R, zc), (XT + 0.02, Y_R, zc)], cap, pl)
        # pedestal for the gourd
        sb.box(-0.07, 0.07, Y_R - 0.06, Y_R + 0.06, zb - 0.01, zc + 0.03, pl)
    with sb.st(grime=0.45, smooth=True):
        g = [(0.03, 0.0), (0.042, 0.02), (0.045, 0.045), (0.034, 0.07), (0.018, 0.082), (0.026, 0.095), (0.028, 0.11),
             (0.02, 0.128), (0.008, 0.137), (0.004, 0.15), (0.0005, 0.152)]
        sb.lathe(g, (0.0, Y_R, zc + 0.03), pl, segs=10)
    # ridge ends: solid plastered blocks over the gables, sweeping up and curling inwards at the tips
    body = [(-0.06, 0.0), (0.06, 0.0), (0.062, 0.07), (0.052, 0.12), (0.03, 0.14), (-0.03, 0.14), (-0.052, 0.12), (-0.062, 0.07)]
    X0, X1, R = 0.50, 0.84, 0.07
    for sx in (-1, 1):
        xs = np.linspace(X0, X1, 10)
        pts = [(sx * x, Y_R, zb - 0.01 + R * ((x - X0) / (X1 - X0)) ** 2.2) for x in xs]
        scl = [(1.0, 1.0)] * len(pts)
        ang = math.atan(R * 2.2 / (X1 - X0))
        x, z, r = X1, pts[-1][2], 0.055
        for k in range(7):
            ang += math.radians(16)
            x += r * 0.4 * math.cos(ang)
            z += r * 0.4 * math.sin(ang)
            r *= 0.92
            pts.append((sx * x, Y_R, z))
            scl.append((max(0.42, 1.0 - 0.08 * (k + 1)), max(0.38, 1.0 - 0.09 * (k + 1))))
        with sb.st(grime=0.55):
            sb.sweep(pts, body, pl, scale=scl)


def _copings(sb, M, rng):
    """Lime-plastered gable copings riding on the tile ends over the ridge, slightly upturned at the eaves."""
    prof = [(-0.072, -0.02), (0.072, -0.02), (0.074, 0.03), (0.062, 0.058), (0.034, 0.074), (-0.034, 0.074),
            (-0.062, 0.058), (-0.074, 0.03)]
    for xc in (-0.738, 0.738):
        path = []
        yf, yb = Y_EF - 0.035, Y_EB + 0.035
        for y in np.concatenate([np.linspace(yf, Y_R, 12), np.linspace(Y_R, yb, 12)[1:]]):
            s = -1 if y < Y_R else 1
            D = _D(s) + 0.035
            t = abs(y - Y_R)
            up = 0.025 * max(0.0, (t - D + 0.14) / 0.14) ** 2
            yy = min(max(y, Y_EF), Y_EB)
            path.append((xc, y, zdeck(yy) + up - abs(y - yy) * TAN))
        with sb.st(grime=0.72):
            sb.sweep(path, prof, M["plaster"], side=(1, 0, 0))


def _walls(sb, M, rng):
    brick = M["brick"]
    # side walls up to the roof deck (hard gables), with stepped brick corbels (chitou) under the front eave
    top = [(y, zdeck(y) - 0.002) for y in np.linspace(Y_BACK, Y_PORCH, 15)]
    outline = [(Y_PORCH, Z_PL - 0.03), (Y_BACK, Z_PL - 0.03)] + top
    for (x0, x1) in ((-HW, -XI), (XI, HW)):
        sb.prism_x(outline, x0, x1, brick)
        zc = zdeck(Y_PORCH - 0.12) - DECK_T
        for k, (dy, z0, z1) in enumerate(((0.04, zc - 0.25, zc - 0.17), (0.08, zc - 0.17, zc - 0.09), (0.12, zc - 0.09, zc))):
            ox = 0.01 * (k + 1)
            xa, xb = (x0 - ox, x1) if x0 < 0 else (x0, x1 + ox)
            if k < 2:
                sb.box(xa, xb, Y_PORCH - dy, Y_PORCH + 0.01, z0, z1, brick)
            else:
                with sb.st(grime=0.35):
                    sb.box(xa, xb, Y_PORCH - dy, Y_PORCH + 0.01, z0, z1, M["plaster"])
    # front (doorway) wall, back wall
    z_fw = zdeck(Y_FRONT) - DECK_T - 0.045
    fw = Wall((-XI, Y_FRONT, 0), (1, 0, 0), (0, -1, 0), 2 * XI, t=T_FRONT)
    fw.build(sb, Z_PL - 0.03, z_fw, [(XI - DOOR_HW, XI + DOOR_HW, Z_SILL, Z_LINTEL)], brick)
    z_bw = zdeck(Y_BACK) - DECK_T - 0.045
    sb.box(-XI, XI, Y_BACK - T_BACK, Y_BACK, Z_PL - 0.03, z_bw, brick)
    # sooty plaster lining of the niche (inside faces, ceiling) and its floor
    yi0, yi1 = Y_FRONT + T_FRONT + 0.003, Y_BACK - T_BACK - 0.003
    xi = XI - 0.003
    zc0 = zdeck(yi0) - DECK_T - 0.003
    zc1 = zdeck(yi1) - DECK_T - 0.003
    pl = M["plaster"]
    with sb.st(grime=0.85):
        for x in (-xi, xi):
            sb.poly([(x, yi0, Z_SILL), (x, yi1, Z_SILL), (x, yi1, zc1), (x, Y_R, zdeck(Y_R) - DECK_T - 0.003), (x, yi0, zc0)], pl)
        sb.poly([(-xi, yi1, Z_SILL), (xi, yi1, Z_SILL), (xi, yi1, zc1), (-xi, yi1, zc1)], pl)
        for (ya, yb) in ((yi0, Y_R), (Y_R, yi1)):
            sb.poly([(-xi, ya, zdeck(ya) - DECK_T - 0.003), (xi, ya, zdeck(ya) - DECK_T - 0.003),
                     (xi, yb, zdeck(yb) - DECK_T - 0.003), (-xi, yb, zdeck(yb) - DECK_T - 0.003)], pl)
        y = yi0
        for (xa, xb, za, zb) in ((-xi, -DOOR_HW, Z_SILL, zc0), (DOOR_HW, xi, Z_SILL, zc0), (-DOOR_HW, DOOR_HW, Z_LINTEL, zc0)):
            sb.poly([(xa, y, za), (xb, y, za), (xb, y, zb), (xa, y, zb)], pl)
    with sb.st(grime=0.7):
        sb.box(-XI, XI, Y_FRONT + T_FRONT - 0.01, Y_BACK - T_BACK + 0.01, Z_PL - 0.03, Z_SILL, pl)


def _stonework(sb, M, rng):
    with sb.st(tvar=0.5):
        _stonework_(sb, M, rng)


def _stonework_(sb, M, rng):
    st = M["stone"]
    # plinth (sunk 45 cm so it sits on uneven ground) and a low step
    with sb.st(nch=1.0):
        sb.box(-HW - 0.08, HW + 0.08, APRON_Y, Y_BACK + 0.08, -0.45, Z_PL, st)
    sb.box(-0.42, 0.42, APRON_Y - 0.17, APRON_Y, -0.25, 0.135, st)
    # apron: 3 x 2 cement slabs, slightly settled, 8 mm cracks between them
    xs = [-1.0, -0.33, 0.34, 1.0]
    ys = [0.0, 0.40, APRON_Y]
    for i in range(3):
        for j in range(2):
            z = 0.05 + rng.normal(0, 0.005)
            dz = rng.normal(0, 0.004, 4)
            x0, x1 = xs[i] + (0.004 if i else 0), xs[i + 1] - (0.004 if i < 2 else 0)
            y0, y1 = ys[j] + (0.004 if j else 0), ys[j + 1] - (0.004 if j < 1 else 0.0)
            with sb.st(grime=0.25 + 0.25 * rng.random(), tvar=rng.random()):
                sb.hexa([(x0, y0, -0.25), (x1, y0, -0.25), (x1, y1, -0.25), (x0, y1, -0.25),
                         (x0, y0, z + dz[0]), (x1, y0, z + dz[1]), (x1, y1, z + dz[2]), (x0, y1, z + dz[3])], st)
    # granite door frame: sill (offering ledge), jambs, lintel
    sb.box(-0.31, 0.31, Y_FRONT - 0.045, Y_FRONT + T_FRONT, Z_SILL - 0.06, Z_SILL, st)
    for sx in (-1, 1):
        sb.box(*sorted((sx * DOOR_HW, sx * (DOOR_HW + 0.07))), Y_FRONT - 0.018, Y_FRONT + 0.04, Z_SILL, Z_LINTEL, st)
    sb.box(-0.31, 0.31, Y_FRONT - 0.025, Y_FRONT + 0.04, Z_LINTEL, Z_LINTEL + 0.075, st)
    # altar block in the porch with a slab top (soot, wax and ash on it)
    sb.box(-0.32, 0.32, Y_PORCH + 0.03, Y_FRONT, Z_PL - 0.01, Z_ALTAR - 0.035, st)
    with sb.st(grime=0.6):
        sb.box(-0.36, 0.36, Y_PORCH + 0.005, Y_FRONT, Z_ALTAR - 0.035, Z_ALTAR, st)


def _paper_strip(sb, M, rng, centre, w, nch, axis, old=False):
    """A pasted paper strip: nch character cells of size w along `axis` ('v' vertical couplet, 'h' banner),
    bubbled and with one corner peeling.  UV: u across (0..1), v along in cell units (-0.25 .. nch + 0.25)."""
    nu, nv = 3, int(2 * nch + 3)
    L = (nch + 0.5) * w
    peel = rng.integers(4)
    pts, uvs = [], []
    for j in range(nv):
        vv = j / (nv - 1)
        row, urow = [], []
        for i in range(nu):
            uu = i / (nu - 1)
            bub = 0.0015 * math.sin(7 * vv + 3 * uu + rng.random()) * (0.5 + rng.random())
            corner = (vv if peel < 2 else 1 - vv), (uu if peel % 2 else 1 - uu)
            lift = 0.014 * max(0.0, (corner[0] - 0.82) / 0.18) ** 2 * corner[1] if not old else 0.0
            off = (0.0006 + 0.3 * abs(bub)) if old else (0.0032 + abs(bub) + lift)
            if axis == "v":
                p = (centre[0] + (uu - 0.5) * w, Y_FRONT - off, centre[1] + L / 2 - vv * L)
            else:
                p = (centre[0] - L / 2 + vv * L, Y_FRONT - off, centre[1] + (0.5 - uu) * w)
            row.append(p)
            urow.append((uu, vv * (nch + 0.5) - 0.25))
        pts.append(row)
        uvs.append(urow)
    with sb.st(nch=float(nch), tvar=float(rng.random()), grime=1.0 if old else 0.0, smooth=True):
        sb.grid(pts, M["paper"], uv=uvs)


def _paper_and_cloth(sb, M, rng):
    w = 0.12
    for sx in (-1, 1):
        xc = sx * 0.42
        _paper_strip(sb, M, rng, (xc + rng.normal(0, 0.006), 0.785 + rng.normal(0, 0.01)), w + 0.012, 5, "v", old=True)
        _paper_strip(sb, M, rng, (xc, 0.785), w, 5, "v")
    _paper_strip(sb, M, rng, (rng.normal(0, 0.01), Z_LINTEL + 0.075 + 0.075), 0.118, 4, "h", old=True)
    _paper_strip(sb, M, rng, (0.0, Z_LINTEL + 0.075 + 0.07), 0.105, 4, "h")
    # red cloth swag nailed across the lintel, with two tails
    y0 = Y_FRONT - 0.03
    nu, nv = 25, 6
    P = np.zeros((nv, nu, 3))
    for j in range(nv):
        v = j / (nv - 1)
        for i in range(nu):
            u = i / (nu - 1)
            x = -0.33 + 0.66 * u
            ztop = Z_LINTEL + 0.06 - 0.03 * math.sin(math.pi * u)
            depth = 0.05 + 0.085 * math.sin(math.pi * u) ** 0.8
            fold = (0.011 * math.sin(2 * math.pi * 5.5 * u + 0.6) + 0.005 * math.sin(2 * math.pi * 13 * u + 2.0)) * (0.3 + v)
            P[j, i] = (x + 0.012 * (u - 0.5) * v, y0 - 0.012 * v * math.sin(math.pi * u) - fold - 0.004,
                       ztop - v * depth + 0.012 * v * math.sin(2 * math.pi * 3 * u))
    with sb.st(smooth=True):
        sb.grid(P, M["cloth"])
        for sx in (-1, 1):
            nr = 9
            T = np.zeros((nr, 3, 3))
            for j in range(nr):
                v = j / (nr - 1)
                for i in range(3):
                    u = i / 2
                    tw = 0.5 * v
                    xx = sx * (0.335 - 0.01 * v) + (u - 0.5) * 0.06 * math.cos(tw)
                    T[j, i] = (xx, y0 - 0.006 - (u - 0.5) * 0.06 * math.sin(tw) * sx - 0.004 * math.sin(5 * v),
                               Z_LINTEL + 0.065 - v * (0.30 + 0.04 * sx))
            sb.grid(T, M["cloth"])
        # knots at the nails
        for sx in (-1, 1):
            sb.lathe([(0.004, -0.018), (0.02, -0.012), (0.024, 0.0), (0.018, 0.014), (0.003, 0.018)],
                     (sx * 0.335, y0 - 0.012, Z_LINTEL + 0.055), M["cloth"], segs=8)


def _niche(sb, M, rng):
    """Dim interior: red pedestal, seated painted Tudi Gong with a red cape, spirit tablet, small burner, cups."""
    sm = M["smalls"]
    yc = 1.60
    with sb.st(col=(0.20, 0.035, 0.025), rgh=0.55):
        sb.box(-0.12, 0.12, yc - 0.10, yc + 0.11, Z_SILL, Z_SILL + 0.075, sm)
    z0 = Z_SILL + 0.075
    with sb.st(col=(0.26, 0.15, 0.035), rgh=0.45, smooth=True):                 # ochre painted robe (lap)
        sb.lathe([(0.075, 0.0), (0.088, 0.03), (0.084, 0.07), (0.066, 0.095), (0.05, 0.10)], (0, yc, z0), sm, segs=12, cap_bot=True)
    with sb.st(smooth=True):                                                     # red cape over the shoulders
        sb.lathe([(0.064, 0.085), (0.068, 0.13), (0.058, 0.19), (0.036, 0.225), (0.016, 0.232)], (0, yc + 0.005, z0), M["cloth"], segs=12)
    with sb.st(col=(0.50, 0.32, 0.19), rgh=0.35, smooth=True):                  # face
        sb.lathe([(0.0005, -0.033), (0.02, -0.027), (0.031, -0.01), (0.032, 0.006), (0.024, 0.024), (0.0005, 0.033)],
                 (0, yc - 0.006, z0 + 0.262), sm, segs=10)
    with sb.st(col=(0.70, 0.69, 0.64), rgh=0.5, smooth=True):                   # long white beard
        sb.lathe([(0.022, 0.0), (0.016, -0.03), (0.006, -0.07), (0.0005, -0.085)], (0, yc - 0.03, z0 + 0.25), sm, segs=8)
    with sb.st(col=(0.025, 0.022, 0.02), rgh=0.4, smooth=True):                 # official's hat with wings
        sb.lathe([(0.032, 0.0), (0.033, 0.022), (0.022, 0.036), (0.0005, 0.04)], (0, yc - 0.004, z0 + 0.285), sm, segs=10)
    with sb.st(col=(0.025, 0.022, 0.02), rgh=0.4):
        sb.box(-0.075, 0.075, yc + 0.005, yc + 0.012, z0 + 0.295, z0 + 0.305, sm)
    for sx in (-1, 1):                                                           # sleeves and hands on the knees
        sh = (sx * 0.056, yc - 0.005, z0 + 0.195)
        wr = (sx * 0.032, yc - 0.078, z0 + 0.112)
        with sb.st(col=(0.26, 0.15, 0.035), rgh=0.45):
            sb.cyl(sh, wr, 0.021, sm, segs=8)
        with sb.st(col=(0.50, 0.32, 0.19), rgh=0.35, smooth=True):
            sb.lathe([(0.0005, -0.012), (0.012, -0.006), (0.013, 0.004), (0.0005, 0.01)],
                     (wr[0], wr[1] - 0.012, wr[2] - 0.004), sm, segs=8)
    with sb.st(col=(0.08, 0.05, 0.03), rgh=0.6):                                # staff
        sb.cyl((0.085, yc - 0.03, z0 - 0.07), (0.095, yc - 0.035, z0 + 0.30), 0.006, sm, segs=6)
    # red paper spirit tablet on the back wall
    yt = Y_BACK - T_BACK - 0.005
    w = 0.12
    L = 3.5 * w
    with sb.st(nch=3.0, tvar=float(rng.random()), smooth=False):
        q = [(-w / 2, yt, 1.02 - L), (w / 2, yt, 1.02 - L), (w / 2, yt, 1.02), (-w / 2, yt, 1.02)]
        sb.poly(q, M["paper"], uv=[(0, 3.25), (1, 3.25), (1, -0.25), (0, -0.25)])
    # small inner burner with stubs, two cups
    _burner(sb, M, rng, (0.0, 1.40, Z_SILL), 0.42, n_stubs=6, n_lit=0)
    for x in (-0.13, 0.13):
        _cup(sb, M, (x, 1.42, Z_SILL), 0.9)


def _cup(sb, M, c, s=1.0):
    prof = [(0.016, 0.0), (0.019, 0.004), (0.025, 0.024), (0.0265, 0.028), (0.024, 0.027), (0.015, 0.007), (0.0005, 0.006)]
    with sb.st(col=(0.74, 0.73, 0.69), rgh=0.12, smooth=True):
        sb.lathe([(r * s, z * s) for r, z in prof], c, M["smalls"], segs=12, cap_bot=True)
    with sb.st(col=(0.10, 0.06, 0.018), rgh=0.03):
        sb.lathe([(0.0005, 0.019 * s), (0.0215 * s, 0.019 * s)], c, M["smalls"], segs=12)


def _burner(sb, M, rng, c, s=1.0, n_stubs=22, n_lit=3):
    """Bronze incense burner on three feet, full of ash, red incense stubs and a few lit sticks."""
    sm = M["smalls"]
    cx, cy, cz = c
    feet = 0.035 * s
    prof = [(0.055, 0.0), (0.08, 0.012), (0.095, 0.035), (0.10, 0.065), (0.098, 0.085), (0.106, 0.092), (0.108, 0.10),
            (0.098, 0.102), (0.094, 0.088)]
    with sb.st(col=(0.085, 0.065, 0.04), rgh=0.42, met=0.85, smooth=True):
        sb.lathe([(r * s, z * s) for r, z in prof], (cx, cy, cz + feet), sm, segs=16, cap_bot=True)
    with sb.st(col=(0.085, 0.065, 0.04), rgh=0.42, met=0.85):
        for k in range(3):
            a = 2 * math.pi * k / 3 + math.pi / 2
            sb.cyl((cx + 0.06 * s * math.cos(a), cy + 0.06 * s * math.sin(a), cz),
                   (cx + 0.055 * s * math.cos(a), cy + 0.055 * s * math.sin(a), cz + feet + 0.01), 0.011 * s, sm, segs=6)
        for sx in (-1, 1):                                                       # upright ears on the rim
            x = cx + sx * 0.098 * s
            sb.box(x - 0.006 * s, x + 0.006 * s, cy - 0.022 * s, cy - 0.012 * s, cz + feet + 0.1 * s, cz + feet + 0.135 * s, sm)
            sb.box(x - 0.006 * s, x + 0.006 * s, cy + 0.012 * s, cy + 0.022 * s, cz + feet + 0.1 * s, cz + feet + 0.135 * s, sm)
            sb.box(x - 0.006 * s, x + 0.006 * s, cy - 0.022 * s, cy + 0.022 * s, cz + feet + 0.127 * s, cz + feet + 0.137 * s, sm)
    za = cz + feet + 0.085 * s
    with sb.st(col=(0.30, 0.29, 0.27), rgh=0.95):                               # ash
        a = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        sb.poly([(cx + 0.094 * s * math.cos(t), cy + 0.094 * s * math.sin(t), za + 0.003 * math.sin(3 * t)) for t in a], sm)
    for k in range(n_stubs):                                                     # burnt-down red stubs
        r = 0.085 * s * math.sqrt(rng.random())
        t = rng.uniform(0, 2 * math.pi)
        p0 = np.array([cx + r * math.cos(t), cy + r * math.sin(t), za - 0.01])
        h = rng.uniform(0.02, 0.075) * (0.6 + 0.4 * s)
        d = np.array([rng.normal(0, 0.12), rng.normal(0, 0.12), 1.0])
        d /= np.linalg.norm(d)
        p1 = p0 + d * h
        with sb.st(col=(0.42 + 0.1 * rng.random(), 0.03, 0.025), rgh=0.6):
            sb.cyl(p0, p1, 0.0018, sm, segs=4)
        with sb.st(col=(0.03, 0.025, 0.022), rgh=0.9):
            sb.cyl(p1, p1 + d * 0.004, 0.0019, sm, segs=4)
    for k in range(n_lit):                                                       # three lit sticks
        t = (k - (n_lit - 1) / 2) * 0.35
        p0 = np.array([cx + 0.025 * math.sin(t), cy + 0.01 * k - 0.01, za - 0.01])
        d = np.array([math.sin(t) * 0.18, rng.normal(0, 0.05), 1.0])
        d /= np.linalg.norm(d)
        p1 = p0 + d * 0.085
        p2 = p1 + d * rng.uniform(0.13, 0.19)
        with sb.st(col=(0.50, 0.035, 0.03), rgh=0.55):
            sb.cyl(p0, p1, 0.0017, sm, segs=5)
        with sb.st(col=(0.16, 0.10, 0.06), rgh=0.85):
            sb.cyl(p1, p2, 0.0026, sm, segs=5)
        with sb.st(col=(0.42, 0.40, 0.37), rgh=0.95):
            sb.cyl(p2, p2 + d * 0.006, 0.0027, sm, segs=5)
        with sb.st(col=(1.0, 0.32, 0.06), rgh=0.9, emit=6.0):
            sb.cyl(p2 + d * 0.006, p2 + d * 0.009, 0.0024, sm, segs=5)


def _offerings(sb, M, rng):
    sm = M["smalls"]
    _burner(sb, M, rng, (0.0, Y_PORCH + 0.13, Z_ALTAR))
    for sx, h in ((-1, 0.095), (1, 0.055)):                                    # red candle stubs in their wax
        c = (sx * 0.27, Y_PORCH + 0.18, Z_ALTAR)
        with sb.st(col=(0.36, 0.016, 0.014), rgh=0.3, smooth=True):
            sb.lathe([(0.04, 0.0), (0.036, 0.004), (0.022, 0.008), (0.0205, 0.012), (0.0205, h - 0.006), (0.019, h),
                      (0.012, h - 0.004), (0.0005, h - 0.006)], c, sm, segs=12)
            for k in range(3):
                a = rng.uniform(0, 2 * math.pi)
                x, y = c[0] + 0.021 * math.cos(a), c[1] + 0.021 * math.sin(a)
                sb.box(x - 0.004, x + 0.004, y - 0.004, y + 0.004, Z_ALTAR + 0.004, Z_ALTAR + rng.uniform(0.03, h), sm)
        with sb.st(col=(0.02, 0.018, 0.016), rgh=0.9):
            sb.cyl((c[0], c[1], Z_ALTAR + h - 0.008), (c[0] + 0.002, c[1], Z_ALTAR + h + 0.008), 0.0012, sm, segs=4)
    for k, (dx, dy, dz) in enumerate(((-0.195, 0.05, 0.0), (-0.135, 0.035, 0.0), (-0.165, 0.042, 0.048))):   # mandarins
        c = (dx, Y_PORCH + dy, Z_ALTAR + 0.027 + dz)
        col = (0.56, 0.17 + 0.04 * rng.random(), 0.015)
        with sb.st(col=col, rgh=0.42, smooth=True):
            prof = [(0.033 * math.sin(t), -0.033 * math.cos(t)) for t in np.linspace(0.05, math.pi - 0.05, 7)]
            sb.lathe(prof, c, sm, segs=10, sz=0.82)
        with sb.st(col=(0.05, 0.08, 0.02), rgh=0.6):
            sb.box(c[0] - 0.003, c[0] + 0.003, c[1] - 0.003, c[1] + 0.003, c[2] + 0.025, c[2] + 0.031, sm)
    with sb.st(col=(0.48, 0.36, 0.13), rgh=0.85):                               # bundle of joss paper
        sb.box(0.12, 0.215, Y_PORCH + 0.07, Y_PORCH + 0.135, Z_ALTAR, Z_ALTAR + 0.028, sm)
    with sb.st(col=(0.45, 0.04, 0.03), rgh=0.8):
        sb.box(0.16, 0.175, Y_PORCH + 0.068, Y_PORCH + 0.137, Z_ALTAR + 0.001, Z_ALTAR + 0.029, sm)
    for x in (-0.11, 0.0, 0.11):                                                # rice wine cups on the sill
        _cup(sb, M, (x, Y_FRONT + 0.005, Z_SILL), 1.0)


def _surface_z(x, y):
    """Top of whatever is under (x, y): apron slabs, step, or the ground."""
    if abs(x) < 1.0 and 0.0 < y < APRON_Y:
        return 0.05
    return 0.0


def _debris(sb, M, rng):
    """Firecracker shells, burnt red paper and a black ash patch where paper money was burnt."""
    sm = M["smalls"]
    ax, ay = 1.32, 0.42                                                         # ash patch beside the apron
    a = np.linspace(0, 2 * math.pi, 22, endpoint=False)
    rr = 0.27 * (1 + 0.25 * np.sin(3 * a + 1) + 0.15 * rng.normal(size=a.size))
    with sb.st(col=(0.022, 0.02, 0.019), rgh=0.95):
        sb.poly([(ax + r * math.cos(t), ay + 0.8 * r * math.sin(t), 0.004) for r, t in zip(rr, a)], sm)
    def scrap(x, y, z, size, col, rgh=0.8):
        t = rng.uniform(0, math.pi)
        c, s_ = math.cos(t), math.sin(t)
        q = []
        for (u, v) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            u, v = u * size * rng.uniform(0.6, 1.0), v * size * 0.7 * rng.uniform(0.5, 1.0)
            q.append((x + u * c - v * s_, y + u * s_ + v * c, z + rng.uniform(0.0, 0.004)))
        with sb.st(col=col, rgh=rgh):
            sb.poly(q, sm)
    def red():
        f = rng.random()
        return (0.38 + 0.15 * f, 0.02 + 0.08 * f, 0.02 + 0.05 * f)
    for k in range(85):                                                         # scraps on the apron and round it
        if k < 30:
            x, y = ax + rng.normal(0, 0.16), ay + rng.normal(0, 0.12)
            col = [(0.02, 0.018, 0.017), (0.30, 0.29, 0.27), (0.42, 0.32, 0.12)][int(rng.choice(3, p=[0.45, 0.35, 0.2]))]
        else:
            x, y = rng.normal(0, 0.55), rng.uniform(-0.35, APRON_Y + 0.05)
            col = red() if rng.random() < 0.7 else (0.025, 0.022, 0.02)
        scrap(x, y, _surface_z(x, y) + 0.001, rng.uniform(0.006, 0.02), col)
    for k in range(32):                                                         # firecracker shells
        x, y = rng.normal(0.0, 0.5), rng.uniform(-0.4, APRON_Y)
        z = _surface_z(x, y)
        t = rng.uniform(0, math.pi)
        r = rng.uniform(0.0035, 0.0055)
        L = rng.uniform(0.02, 0.04)
        p0 = np.array([x, y, z + r])
        p1 = p0 + L * np.array([math.cos(t), math.sin(t), 0.0])
        with sb.st(col=red(), rgh=0.75):
            sb.cyl(p0, p1, r, sm, segs=5, caps=False)
        with sb.st(col=(0.03, 0.025, 0.02), rgh=0.9):
            sb.poly([p0 + r * np.array([math.cos(q) * -math.sin(t), math.cos(q) * math.cos(t), math.sin(q)])
                     for q in np.linspace(0, 2 * math.pi, 5, endpoint=False)], sm)
    for k in range(5):                                                          # wax drips on the step and apron
        x, y = rng.normal(0, 0.2), rng.uniform(APRON_Y - 0.15, APRON_Y - 0.02)
        with sb.st(col=(0.30, 0.02, 0.015), rgh=0.3):
            sb.cyl((x, y, 0.135), (x, y, 0.139), rng.uniform(0.006, 0.014), sm, segs=7)


def _leaves(sb, M, rng):
    """Dry leaves blown against the shrine: on the apron, the porch floor, the plinth ledges and the roof."""
    sm = M["smalls"]
    spots = []
    for k in range(26):
        spots.append((rng.normal(0, 0.7), rng.uniform(-0.6, APRON_Y), None))
    for k in range(6):
        spots.append((rng.uniform(-XI + 0.02, XI - 0.02), rng.uniform(Y_PORCH + 0.01, Y_PORCH + 0.03), Z_PL))
    for k in range(8):
        sx = rng.choice([-1, 1])
        spots.append((sx * rng.uniform(HW + 0.01, HW + 0.07), rng.uniform(APRON_Y + 0.02, Y_BACK), Z_PL))
    for k in range(5):
        x = rng.choice(np.arange(-0.60, 0.601, ROW)) + rng.normal(0, 0.02)
        y = rng.uniform(Y_EF + 0.05, Y_R - 0.15)
        spots.append((x, y, _RP(x, y, T_T + 0.002)[2]))
    for (x, y, z) in spots:
        if z is None:
            z = _surface_z(x, y)
        L, W = rng.uniform(0.035, 0.075), rng.uniform(0.012, 0.025)
        t = rng.uniform(0, 2 * math.pi)
        d = np.array([math.cos(t), math.sin(t), 0.0])
        n = np.array([-math.sin(t), math.cos(t), 0.0])
        c = np.array([x, y, z + 0.002])
        curl = rng.uniform(0.0, 0.012)
        f = rng.random()
        col = [(0.14, 0.075, 0.03), (0.26, 0.17, 0.05), (0.07, 0.045, 0.025)][int(rng.choice(3, p=[0.45, 0.3, 0.25]))]
        col = tuple(v * (0.8 + 0.4 * f) for v in col)
        with sb.st(col=col, rgh=0.75):
            a0, a1 = c - d * L / 2, c + d * L / 2 + np.array([0, 0, rng.uniform(0, 0.01)])
            m0, m1 = c - n * W / 2 + np.array([0, 0, curl]), c + n * W / 2 + np.array([0, 0, curl])
            sb.poly([a0, m0, a1], sm)
            sb.poly([a0, a1, m1], sm)


def _weeds(sb, M, rng):
    sm = M["smalls"]
    spots = []
    for k in range(26):                                                         # round the plinth and apron edges
        edge = rng.integers(4)
        if edge == 0:
            p = (rng.uniform(-1.05, 1.05), rng.uniform(-0.08, -0.01), 0.0)
        elif edge == 1:
            p = (rng.choice([-1, 1]) * rng.uniform(1.0, 1.06), rng.uniform(0.0, APRON_Y), 0.0)
        else:
            sx = rng.choice([-1, 1])
            p = (sx * rng.uniform(HW + 0.08, HW + 0.15), rng.uniform(APRON_Y, Y_BACK + 0.1), 0.0)
        spots.append((p, rng.uniform(0.08, 0.28), int(rng.integers(6, 14))))
    for k in range(8):                                                          # behind the plinth
        spots.append(((rng.uniform(-1.0, 1.0), rng.uniform(Y_BACK + 0.09, Y_BACK + 0.16), 0.0), rng.uniform(0.1, 0.3),
                      int(rng.integers(6, 14))))
    for k in range(7):                                                          # in the joint of wall and plinth
        if rng.random() < 0.7:
            p = (rng.choice([-1, 1]) * rng.uniform(HW + 0.015, HW + 0.05), rng.uniform(Y_PORCH + 0.05, Y_BACK), Z_PL)
        else:
            p = (rng.uniform(-HW, HW), rng.uniform(Y_BACK + 0.015, Y_BACK + 0.05), Z_PL)
        spots.append((p, rng.uniform(0.04, 0.12), int(rng.integers(3, 8))))
    for k in range(4):                                                          # in the cracks of the apron
        p = (rng.choice([-0.33, 0.34]) + rng.normal(0, 0.003), rng.uniform(0.05, APRON_Y - 0.05), 0.04)
        spots.append((p, rng.uniform(0.04, 0.09), int(rng.integers(3, 7))))
    for k in range(3):                                                          # on the roof against the copings
        s = rng.choice([-1, 1])
        y = Y_R + s * rng.uniform(0.25, 0.5)
        x = rng.choice([-1, 1]) * 0.61
        spots.append((_RP(x, y, 0.04), rng.uniform(0.05, 0.10), int(rng.integers(4, 8))))
    for (p, hmax, nb) in spots:
        dry = rng.random() < 0.3
        for b in range(nb):
            bx, by = p[0] + rng.normal(0, 0.02), p[1] + rng.normal(0, 0.02)
            az = rng.uniform(0, 2 * math.pi)
            lean = rng.uniform(0.15, 0.9)
            h = hmax * rng.uniform(0.45, 1.0)
            w = rng.uniform(0.003, 0.0055)
            d = np.array([math.cos(az), math.sin(az), 0.0])
            sdir = np.array([-math.sin(az), math.cos(az), 0.0])
            base = np.array([bx, by, p[2] - 0.01])
            mid = base + d * h * 0.35 * lean + np.array([0, 0, h * 0.55])
            tip = base + d * h * 1.0 * lean + np.array([0, 0, h * (0.95 - 0.35 * lean)])
            f = rng.random()
            col = (0.16 + 0.1 * f, 0.15 + 0.05 * f, 0.07) if (dry and rng.random() < 0.7) else (0.035 + 0.04 * f, 0.07 + 0.05 * f, 0.018)
            with sb.st(col=col, rgh=0.55):
                sb.poly([base - sdir * w, base + sdir * w, mid + sdir * w * 0.8, mid - sdir * w * 0.8], sm)
                sb.poly([mid - sdir * w * 0.8, mid + sdir * w * 0.8, tip], sm)


# ------------------------------------------------------------------ materials
class _G:
    """Tiny node-graph helper: inputs may be sockets or constants."""

    def __init__(self, nt):
        self.nt = nt

    def _in(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif v is not None:
            dv = sock.default_value
            if hasattr(dv, "__len__"):
                if isinstance(v, (int, float)):
                    v = (v,) * len(dv)
                elif len(v) == 3 and len(dv) == 4:
                    v = (*v, 1.0)
            sock.default_value = v

    def n(self, typ, ins=None, **props):
        nd = self.nt.nodes.new(typ)
        for k, v in props.items():
            setattr(nd, k, v)
        for k, v in (ins or {}).items():
            self._in(nd.inputs[k], v)
        return nd

    def math(self, op, a, b=0.0, c=None, clamp=False):
        nd = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._in(nd.inputs[0], a)
        self._in(nd.inputs[1], b)
        if c is not None:
            self._in(nd.inputs[2], c)
        return nd.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        nd = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self._in(nd.inputs["Factor"], fac)
        self._in(nd.inputs["A"], a)
        self._in(nd.inputs["B"], b)
        return nd.outputs[2]

    def ramp(self, x, e0, e1, lo=0.0, hi=1.0):
        nd = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        for i, v in enumerate((x, e0, e1, lo, hi)):
            self._in(nd.inputs[i], v)
        return nd.outputs[0]

    def noise(self, vec, scale, detail=2.0, rough=0.5, w=None, dist=0.0):
        nd = self.n("ShaderNodeTexNoise", noise_dimensions="4D" if w is not None else "3D")
        self._in(nd.inputs["Vector"], vec)
        self._in(nd.inputs["Scale"], scale)
        self._in(nd.inputs["Detail"], detail)
        self._in(nd.inputs["Roughness"], rough)
        self._in(nd.inputs["Distortion"], dist)
        if w is not None:
            self._in(nd.inputs["W"], w)
        return nd.outputs[0]

    def vmul(self, v, s):
        nd = self.n("ShaderNodeVectorMath", operation="MULTIPLY")
        self._in(nd.inputs[0], v)
        self._in(nd.inputs[1], s)
        return nd.outputs[0]

    def vadd(self, v, s):
        nd = self.n("ShaderNodeVectorMath", operation="ADD")
        self._in(nd.inputs[0], v)
        self._in(nd.inputs[1], s)
        return nd.outputs[0]

    def sep(self, v):
        nd = self.n("ShaderNodeSeparateXYZ")
        self._in(nd.inputs[0], v)
        return nd.outputs

    def comb(self, x, y, z):
        nd = self.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self._in(nd.inputs[i], v)
        return nd.outputs[0]

    def gray(self, v):
        nd = self.n("ShaderNodeCombineColor")
        for i in range(3):
            self._in(nd.inputs[i], v)
        return nd.outputs[0]

    def bw(self, c):
        nd = self.n("ShaderNodeRGBToBW")
        self._in(nd.inputs[0], c)
        return nd.outputs[0]

    def attr(self, name, out="Factor"):
        return self.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name=name).outputs[out]

    def coords(self):
        tc = self.n("ShaderNodeTexCoord")
        oi = self.n("ShaderNodeObjectInfo")
        off = self.vmul(self.gray(oi.outputs["Random"]), (13.7, 7.3, 5.1))
        return tc.outputs["Object"], self.vadd(tc.outputs["Object"], off), tc.outputs["Normal"], tc.outputs["UV"]

    def img(self, path, vec, size_m, noncolor=False, blend=0.25):
        """Box-projected image at its real-world size (procedural stand-in if the asset is missing)."""
        v = self.vmul(vec, 1.0 / size_m)
        if not os.path.exists(path):
            return self.n("ShaderNodeTexNoise", {"Vector": v, "Scale": 6.0, "Detail": 8.0}).outputs["Color"]
        im = bpy.data.images.load(path, check_existing=True)
        if noncolor:
            im.colorspace_settings.name = "Non-Color"
        nd = self.n("ShaderNodeTexImage", projection="BOX", projection_blend=blend, interpolation="Cubic")
        nd.image = im
        self._in(nd.inputs[0], v)
        return nd.outputs["Color"]


def _new(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(p.outputs[0], out.inputs[0])
    return m, _G(nt), p, out


def _streaks(g, co, z_lo, z_hi):
    """Rain streaks: vertically stretched noise, strongest just under the roof edge."""
    nz = g.noise(g.vmul(co, (6.5, 6.5, 0.3)), 1.0, detail=5.0, rough=0.62)
    st = g.ramp(nz, 0.46, 0.72)
    z = g.sep(co)[2]
    return g.math("MULTIPLY", st, g.ramp(z, z_lo, z_hi))


def _soot(g, co, cot, z0=Z_LINTEL - 0.05):
    """Incense smoke stain: rises from the burner past the lintel, darkest just above the doorway."""
    x, y, z = g.sep(co)
    gx = g.math("EXPONENT", g.math("MULTIPLY", g.math("POWER", g.math("DIVIDE", x, 0.30), 2.0), -1.0))
    gz = g.math("MULTIPLY", g.ramp(z, z0, Z_LINTEL + 0.2), g.math("SUBTRACT", 1.0, g.ramp(z, Z_LINTEL + 0.25, Z_LINTEL + 0.62)))
    fy = g.math("SUBTRACT", 1.0, g.ramp(y, Y_FRONT + 0.16, Y_FRONT + 0.22))
    ns = g.noise(g.vmul(cot, (1.0, 1.0, 0.4)), 6.0, detail=4.0)
    return g.math("MULTIPLY", g.math("MULTIPLY", gx, gz), g.math("MULTIPLY", fy, g.math("MULTIPLY_ADD", ns, 0.9, 0.35)), clamp=True)


def _bump(g, p, height, strength, dist, bevel=0.0):
    b = g.n("ShaderNodeBump", {"Strength": strength, "Distance": dist})
    g._in(b.inputs["Height"], height)
    if bevel:
        bv = g.n("ShaderNodeBevel", {"Radius": bevel}, samples=6)
        g._in(b.inputs["Normal"], bv.outputs[0])
    g._in(p.inputs["Normal"], b.outputs[0])


def _mat_brick(tex):
    m = bpy.data.materials.get("ShrineBrick")
    if m:
        return m
    m, g, p, _ = _new("ShrineBrick")
    co, cot, nrm, _uv = g.coords()
    z = g.sep(co)[2]
    d = g.img(f"{tex}/red_bricks_04/red_bricks_04_diff_2k.jpg", cot, 2.7)
    bwd = g.bw(d)
    lum = g.math("DIVIDE", bwd, 0.262)
    lumc = g.math("MULTIPLY_ADD", lum, 0.7, 0.3)
    col = g.mix(1.0, g.gray(lumc), (0.172, 0.178, 0.186), "MULTIPLY")            # blue-grey fired brick, lime mortar
    ratio = g.n("ShaderNodeVectorMath", operation="DIVIDE")
    g._in(ratio.inputs[0], d)
    g._in(ratio.inputs[1], g.gray(g.math("MAXIMUM", bwd, 0.02)))
    col = g.mix(1.0, col, g.mix(0.09, (1.0, 1.0, 1.0), ratio.outputs[0]), "MULTIPLY")   # brick-to-brick firing tint
    n1 = g.noise(cot, 1.3, detail=3.0)
    col = g.mix(g.ramp(n1, 0.35, 0.7), g.mix(1.0, col, (1.18, 1.05, 0.88), "MULTIPLY"), g.mix(1.0, col, (0.85, 0.9, 0.98), "MULTIPLY"))
    n2 = g.noise(cot, 4.0, detail=4.0)
    col = g.mix(g.math("MULTIPLY", g.ramp(n2, 0.6, 0.75), 0.35), col, (0.26, 0.255, 0.24))   # lime efflorescence
    # damp, dark base with moss / algae, rising unevenly from the plinth
    nb = g.noise(g.vmul(cot, (2.0, 2.0, 1.0)), 2.5, detail=4.0)
    hb = g.math("SUBTRACT", z, g.math("MULTIPLY_ADD", nb, 0.45, Z_PL - 0.12))
    damp = g.math("SUBTRACT", 1.0, g.ramp(hb, 0.0, 0.32))
    col = g.mix(damp, col, g.mix(1.0, col, (0.5, 0.5, 0.48), "MULTIPLY"))
    moss = g.math("MULTIPLY", damp, g.ramp(g.noise(cot, 11.0, detail=6.0, rough=0.65), 0.46, 0.62))
    col = g.mix(moss, col, (0.030, 0.040, 0.016))
    alg = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.ramp(z, 0.3, 1.1)), g.ramp(g.noise(cot, 1.8, detail=5.0), 0.5, 0.68))
    col = g.mix(g.math("MULTIPLY", alg, 0.6), col, g.mix(1.0, col, (0.45, 0.52, 0.38), "MULTIPLY"))   # green-black algae
    # rain streaks from the copings and corbels
    st = _streaks(g, cot, 0.25, 1.6)
    col = g.mix(g.math("MULTIPLY", st, 0.75), col, g.mix(1.0, col, (0.38, 0.40, 0.37), "MULTIPLY"))
    # soot from the incense: above the doorway and inside its reveals, fading up the wall
    soot = g.math("ADD", _soot(g, co, cot), g.attr("grime"), clamp=True)
    col = g.mix(g.math("MULTIPLY", soot, 0.92), col, (0.014, 0.012, 0.011))
    g._in(p.inputs["Base Color"], col)
    r = g.img(f"{tex}/red_bricks_04/red_bricks_04_rough_2k.jpg", cot, 2.7, noncolor=True)
    g._in(p.inputs["Roughness"], g.math("MULTIPLY_ADD", g.bw(r), 0.6, g.math("MULTIPLY_ADD", damp, -0.12, 0.38), clamp=True))
    bh = g.math("SUBTRACT", 1.0, g.math("MINIMUM", lum, 1.6))
    _bump(g, p, bh, 0.55, 0.012, bevel=0.006)
    return m


def _mat_tile(tex):
    m = bpy.data.materials.get("ShrineTile")
    if m:
        return m
    m, g, p, _ = _new("ShrineTile")
    co, cot, nrm, _uv = g.coords()
    tv = g.attr("tvar")
    gr = g.attr("grime")
    d = g.img(f"{tex}/grey_roof_tiles/grey_roof_tiles_diff_2k.jpg", cot, 1.1)
    lum = g.math("DIVIDE", g.bw(d), 0.358)
    v = g.math("MULTIPLY", g.math("MULTIPLY_ADD", g.math("MINIMUM", tv, 1.0), 0.55, 0.72), 0.072)
    col = g.mix(1.0, g.gray(g.math("MULTIPLY", lum, v)), (0.98, 1.0, 1.06), "MULTIPLY")
    col = g.mix(0.3, col, g.mix(1.0, d, (0.22, 0.22, 0.22), "MULTIPLY"))         # a little of the lichen colour
    new = g.ramp(tv, 1.1, 1.3)
    col = g.mix(new, col, g.mix(1.0, g.gray(lum), (0.135, 0.125, 0.115), "MULTIPLY"))
    # algae and dirt in the pan channels, mostly near the eaves
    na = g.noise(cot, 9.0, detail=5.0, rough=0.6)
    alg = g.math("MULTIPLY", gr, g.ramp(na, 0.38, 0.62), clamp=True)
    col = g.mix(g.math("MULTIPLY", alg, 0.85), col, (0.020, 0.026, 0.012))
    # pale dusty lichen crusts on the cover ridges
    nl = g.ramp(g.noise(cot, 22.0, detail=3.0), 0.66, 0.72)
    col = g.mix(g.math("MULTIPLY", nl, g.math("SUBTRACT", 1.0, gr)), col, (0.20, 0.20, 0.17))
    g._in(p.inputs["Base Color"], col)
    g._in(p.inputs["Roughness"], g.math("MULTIPLY_ADD", alg, -0.12, 0.88))
    g._in(p.inputs["Specular IOR Level"], 0.32)
    _bump(g, p, g.math("ADD", g.math("MULTIPLY", lum, 0.5), g.noise(cot, 60.0, detail=3.0)), 0.25, 0.004)
    return m


def _mat_plaster(tex):
    m = bpy.data.materials.get("ShrinePlaster")
    if m:
        return m
    m, g, p, _ = _new("ShrinePlaster")
    co, cot, nrm, _uv = g.coords()
    gr = g.attr("grime")
    d = g.img(f"{tex}/plaster_grey_04/plaster_grey_04_diff_2k.jpg", cot, 1.2)
    lum = g.math("DIVIDE", g.bw(d), 0.56)
    col = g.mix(1.0, g.gray(lum), (0.46, 0.455, 0.43), "MULTIPLY")              # weathered lime wash
    up = g.ramp(g.sep(nrm)[2], 0.2, 0.9)
    nm = g.noise(cot, 5.0, detail=6.0, rough=0.62)
    mould = g.ramp(g.math("ADD", g.math("ADD", nm, g.math("MULTIPLY", up, 0.2)), g.math("MULTIPLY", gr, 0.3)), 0.5, 0.74)
    col = g.mix(g.math("MULTIPLY", mould, 0.92), col, (0.04, 0.042, 0.036))   # black mould on the lime
    st = _streaks(g, cot, 0.3, 2.2)
    col = g.mix(g.math("MULTIPLY", st, 0.5), col, g.mix(1.0, col, (0.35, 0.36, 0.34), "MULTIPLY"))
    soot = g.ramp(gr, 0.6, 1.0)
    col = g.mix(g.math("MULTIPLY", soot, g.math("MULTIPLY_ADD", g.noise(cot, 3.0), 0.4, 0.75), clamp=True), col, (0.016, 0.014, 0.012))
    g._in(p.inputs["Base Color"], col)
    g._in(p.inputs["Roughness"], 0.88)
    _bump(g, p, lum, 0.35, 0.006, bevel=0.004)
    return m


def _mat_stone(tex):
    m = bpy.data.materials.get("ShrineStone")
    if m:
        return m
    m, g, p, _ = _new("ShrineStone")
    co, cot, nrm, _uv = g.coords()
    z = g.sep(co)[2]
    gr = g.attr("grime")
    d = g.img(f"{tex}/concrete_wall_008/concrete_wall_008_diff_2k.jpg", cot, 1.6)
    lum = g.math("DIVIDE", g.bw(d), 0.522)
    sp = g.noise(cot, 350.0, detail=1.0)                                        # granite speckle
    lum = g.math("MULTIPLY", lum, g.math("MULTIPLY_ADD", g.ramp(sp, 0.3, 0.7), 0.35, 0.82))
    col = g.mix(1.0, g.gray(lum), (0.19, 0.186, 0.176), "MULTIPLY")
    col = g.mix(1.0, col, g.gray(g.math("MULTIPLY_ADD", g.attr("tvar"), 0.35, 0.82)), "MULTIPLY")   # slab to slab
    x, y = g.sep(co)[0], g.sep(co)[1]
    up = g.ramp(g.sep(nrm)[2], 0.5, 0.9)
    nd2 = g.noise(cot, 7.0, detail=4.0)
    drip = None
    for dist in (g.math("ABSOLUTE", g.math("SUBTRACT", y, Y_EF - 0.03)), g.math("ABSOLUTE", g.math("SUBTRACT", y, Y_EB + 0.03)),
                 g.math("ABSOLUTE", g.math("SUBTRACT", g.math("ABSOLUTE", x), 0.80))):
        b = g.math("SUBTRACT", 1.0, g.ramp(dist, 0.0, g.math("MULTIPLY_ADD", nd2, 0.12, 0.02)))
        drip = b if drip is None else g.math("MAXIMUM", drip, b)
    drip = g.math("MULTIPLY", drip, up)                                         # eave drip lines: dark, algae
    col = g.mix(g.math("MULTIPLY", drip, 0.75), col, g.mix(1.0, col, (0.35, 0.38, 0.30), "MULTIPLY"))
    dust = g.math("MULTIPLY", up, g.ramp(g.noise(cot, 2.5, detail=5.0), 0.4, 0.7))
    col = g.mix(g.math("MULTIPLY", dust, 0.45), col, (0.13, 0.11, 0.085))       # dust and soil tracked onto the slabs
    ground = g.math("SUBTRACT", 1.0, g.ramp(z, -0.02, 0.2))                    # soil splash at the foot
    col = g.mix(g.math("MULTIPLY", ground, 0.7), col, (0.06, 0.05, 0.037))
    nd = g.noise(cot, 4.0, detail=5.0)
    col = g.mix(g.math("MULTIPLY", gr, g.ramp(nd, 0.25, 0.6), clamp=True), col, (0.03, 0.026, 0.022))   # wax, soot, wet
    moss = g.math("MULTIPLY", ground, g.ramp(g.noise(cot, 14.0, detail=6.0), 0.48, 0.62))
    col = g.mix(moss, col, (0.022, 0.032, 0.010))
    jt = None                                                                   # granite block joints on the plinth
    for (c, pos) in ((x, -0.47), (x, 0.08), (x, 0.61), (y, 1.02), (y, 1.5)):
        j = g.math("SUBTRACT", 1.0, g.ramp(g.math("ABSOLUTE", g.math("SUBTRACT", c, pos)), 0.002, 0.007))
        jt = j if jt is None else g.math("MAXIMUM", jt, j)
    jt = g.math("MULTIPLY", jt, g.math("GREATER_THAN", g.attr("nch"), 0.5))
    col = g.mix(g.math("MULTIPLY", jt, 0.85), col, (0.035, 0.032, 0.028))
    col = g.mix(g.math("MULTIPLY", _soot(g, co, cot, Z_LINTEL - 0.25), 0.85), col, (0.016, 0.014, 0.012))
    g._in(p.inputs["Base Color"], col)
    r = g.img(f"{tex}/concrete_wall_008/concrete_wall_008_rough_2k.jpg", cot, 1.6, noncolor=True)
    g._in(p.inputs["Roughness"], g.math("MULTIPLY_ADD", g.bw(r), 0.3, 0.62, clamp=True))
    _bump(g, p, g.math("SUBTRACT", g.math("ADD", lum, g.math("MULTIPLY", sp, 0.3)), g.math("MULTIPLY", jt, 2.0)), 0.35, 0.006,
          bevel=0.01)
    return m


def _mat_paper():
    """Red couplet paper: brush characters per cell, faded, water-stained, torn edges (transparent)."""
    m = bpy.data.materials.get("ShrinePaper")
    if m:
        return m
    m, g, p, out = _new("ShrinePaper")
    co, cot, nrm, uv = g.coords()
    u, v, _ = g.sep(uv)
    nch = g.attr("nch")
    old = g.attr("grime")
    seed = g.math("MULTIPLY", g.attr("tvar"), 37.0)
    cell = g.math("FLOOR", v)
    fv = g.math("SUBTRACT", v, cell)
    w = g.math("MULTIPLY_ADD", cell, 3.71, seed)
    inside = g.math("MULTIPLY", g.math("MULTIPLY", g.math("GREATER_THAN", v, 0.0), g.math("LESS_THAN", v, nch)),
                    g.math("MULTIPLY", g.math("MULTIPLY", g.math("GREATER_THAN", u, 0.17), g.math("LESS_THAN", u, 0.83)),
                           g.math("MULTIPLY", g.math("GREATER_THAN", fv, 0.1), g.math("LESS_THAN", fv, 0.9))))
    ink = None
    for (su, sv, wo, th) in ((2.2, 2.2, 0.0, 0.075), (0.8, 5.5, 3.3, 0.085), (5.5, 0.8, 7.7, 0.085)):
        nz = g.noise(g.comb(g.math("MULTIPLY", u, su), g.math("MULTIPLY", fv, sv), 0.0), 1.0, detail=0.5, w=g.math("ADD", w, wo))
        s = g.math("SUBTRACT", 1.0, g.ramp(g.math("ABSOLUTE", g.math("SUBTRACT", nz, 0.5)), th * 0.45, th))
        ink = s if ink is None else g.math("MAXIMUM", ink, s)
    fade_ink = g.ramp(g.noise(g.vmul(uv, (4.0, 1.0, 1.0)), 1.6, detail=3.0, w=seed), 0.3, 0.7, 0.45, 0.95)
    ink = g.math("MULTIPLY", g.math("MULTIPLY", ink, inside), g.math("MULTIPLY", fade_ink, g.math("MULTIPLY_ADD", old, -0.7, 1.0)))
    # paper: vermilion, faded in patches (more on the old layer), water stains low down
    nf = g.noise(cot, 3.5, detail=4.0)
    fade = g.math("ADD", g.math("MULTIPLY", g.ramp(nf, 0.35, 0.75), 0.55), g.math("MULTIPLY", old, 0.55), clamp=True)
    col = g.mix(fade, (0.36, 0.016, 0.014), (0.46, 0.15, 0.115))
    z = g.sep(co)[2]
    stain = g.math("MULTIPLY", g.ramp(g.noise(g.vmul(cot, (2, 2, 0.5)), 4.0, detail=4.0), 0.5, 0.7),
                   g.math("SUBTRACT", 1.0, g.ramp(z, 0.45, 0.8)))
    col = g.mix(g.math("MULTIPLY", stain, 0.6), col, g.mix(1.0, col, (0.45, 0.4, 0.35), "MULTIPLY"))
    col = g.mix(g.math("MULTIPLY", ink, 0.9), col, (0.02, 0.016, 0.016))
    g._in(p.inputs["Base Color"], col)
    g._in(p.inputs["Roughness"], 0.62)
    _bump(g, p, g.noise(uv, 9.0, detail=6.0), 0.15, 0.002)
    # torn edges and missing pieces
    du = g.math("MINIMUM", u, g.math("SUBTRACT", 1.0, u))
    dv = g.math("MINIMUM", g.math("ADD", v, 0.25), g.math("SUBTRACT", g.math("ADD", nch, 0.25), v))
    dv = g.math("MULTIPLY", dv, 1.0)
    edge = g.math("MINIMUM", du, dv)
    nt_ = g.noise(g.vmul(uv, (1.0, 1.0, 1.0)), 7.0, detail=5.0, rough=0.7, w=seed)
    torn = g.math("LESS_THAN", edge, g.math("MULTIPLY_ADD", nt_, g.math("MULTIPLY_ADD", old, 0.18, 0.05), -0.02))
    holes = g.math("GREATER_THAN", g.noise(uv, 1.3, detail=5.0, rough=0.65, w=g.math("ADD", seed, 5.0)),
                   g.math("MULTIPLY_ADD", old, -0.13, 0.69))
    gone = g.math("MAXIMUM", torn, holes)
    tr = g.n("ShaderNodeBsdfTransparent")
    mx = g.n("ShaderNodeMixShader")
    g._in(mx.inputs[0], gone)
    g.nt.links.new(p.outputs[0], mx.inputs[1])
    g.nt.links.new(tr.outputs[0], mx.inputs[2])
    g.nt.links.new(mx.outputs[0], out.inputs[0])
    return m


def _mat_cloth():
    m = bpy.data.materials.get("ShrineCloth")
    if m:
        return m
    m, g, p, _ = _new("ShrineCloth")
    co, cot, nrm, _uv = g.coords()
    n1 = g.noise(cot, 25.0, detail=4.0)
    col = g.mix(g.ramp(n1, 0.3, 0.75), (0.15, 0.006, 0.007), (0.24, 0.011, 0.010))
    up = g.ramp(g.sep(nrm)[2], 0.3, 0.95)
    col = g.mix(g.math("MULTIPLY", up, 0.35), col, (0.16, 0.08, 0.06))          # dust and soot on top of the folds
    g._in(p.inputs["Base Color"], col)
    g._in(p.inputs["Roughness"], 0.8)
    g._in(p.inputs["Sheen Weight"], 0.6)
    g._in(p.inputs["Sheen Roughness"], 0.4)
    _bump(g, p, g.noise(cot, 900.0, detail=1.0), 0.12, 0.001)
    return m


def _mat_smalls():
    """Small props: colour, roughness, metallic and emission come from face attributes."""
    m = bpy.data.materials.get("ShrineSmalls")
    if m:
        return m
    m, g, p, _ = _new("ShrineSmalls")
    co, cot, nrm, _uv = g.coords()
    c = g.attr("col", "Color")
    n1 = g.noise(cot, 45.0, detail=4.0)
    col = g.mix(1.0, c, g.gray(g.math("MULTIPLY_ADD", n1, 0.35, 0.82)), "MULTIPLY")
    g._in(p.inputs["Base Color"], col)
    g._in(p.inputs["Roughness"], g.attr("rgh"))
    g._in(p.inputs["Metallic"], g.attr("met"))
    g._in(p.inputs["Emission Color"], c)
    g._in(p.inputs["Emission Strength"], g.attr("emit"))
    _bump(g, p, g.noise(cot, 220.0, detail=2.0), 0.08, 0.001)
    return m


def materials(assets=None):
    tex = os.path.join(assets or ASSETS, "tex")
    return dict(brick=_mat_brick(tex), tile=_mat_tile(tex), plaster=_mat_plaster(tex), stone=_mat_stone(tex),
                paper=_mat_paper(), cloth=_mat_cloth(), smalls=_mat_smalls())


# ------------------------------------------------------------------ entry point
def earth_god_shrine(coll, x, y, z, heading, rng=None, *, assets=None) -> list:
    """Build one roadside earth-god shrine and link it into `coll`.

    (x, y, z): world position of the front-centre of the paved apron at ground level (the origin); the shrine
    faces `heading` (radians, world angle of the direction it faces, atan2 convention) and its body stands
    0.85-2.0 m behind the origin.  rng: numpy Generator for the per-instance details (tile jitter, paper,
    scraps, weeds).  Returns the list of created objects."""
    rng = rng if rng is not None else np.random.default_rng(7)
    M = materials(assets)
    sb = SB()
    _stonework(sb, M, rng)
    _walls(sb, M, rng)
    _roof(sb, M, rng)
    _ridge(sb, M, rng)
    _copings(sb, M, rng)
    _paper_and_cloth(sb, M, rng)
    _niche(sb, M, rng)
    _offerings(sb, M, rng)
    _debris(sb, M, rng)
    _leaves(sb, M, rng)
    _weeds(sb, M, rng)
    order = list(dict.fromkeys(M.values()))
    idx = {mt.name: i for i, mt in enumerate(order)}
    sb.m = [idx[mt.name] for mt in sb.m]
    me = sb.make("EarthGodShrine", order)
    ob = bpy.data.objects.new("EarthGodShrine", me)
    ob.location = (x, y, z)
    ob.rotation_euler = (0.0, 0.0, heading + math.pi / 2)
    coll.objects.link(ob)
    return [ob]
