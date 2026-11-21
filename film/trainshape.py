"""Parametric CR450AF car-body surface (pure numpy, no Blender).

Each cross-section (half section, right side, y >= 0 mirrored later) is built
from five longitudinal "rails" evaluated at distance X from the nose tip:

  bottom centre (0, zb) -> lower corner (yl, zl) -> crease (yc, zc)
  -> max-width (yw, zw) -> top centre (0, zt)

Segments: S1 bottom (bezier), S2 lower side up to the sharp crease (bezier),
S3 crease -> max width (bezier, forms the bill lip / side blade ledge),
S4 superellipse shoulder/roof with exponent n(X).

The rail values were fitted to a calibrated side photograph (ref08) and
front three-quarter photographs (ref03, ref09); see docs/RESEARCH.md.
"""
import numpy as np

from . import specs as S

# ----------------------------------------------------------------- helpers
def pchip(xk, yk, x):
    """Monotone piecewise cubic Hermite interpolation (Fritsch-Carlson)."""
    xk = np.asarray(xk, float); yk = np.asarray(yk, float); x = np.asarray(x, float)
    h = np.diff(xk); d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0:
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0] = d[0]; m[-1] = d[-1]
    xc = np.clip(x, xk[0], xk[-1])
    i = np.clip(np.searchsorted(xk, xc) - 1, 0, len(h) - 1)
    t = (xc - xk[i]) / h[i]
    h00 = 2 * t**3 - 3 * t**2 + 1; h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2; h11 = t**3 - t**2
    return h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def resample(poly, n):
    """Resample polyline to n points uniformly by arc length."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(poly[:1], n, axis=0)
    st = np.linspace(0, s[-1], n)
    return np.stack([np.interp(st, s, poly[:, k]) for k in range(poly.shape[1])], axis=1)


# ----------------------------------------------------------------- rails (fitted)
# X stations (m from nose tip).  Beyond the last station the section is constant.
RX =   [0.00, 0.06, 0.20, 0.45, 0.80, 1.20, 1.45, 1.75, 2.20, 2.90, 3.80, 4.80, 5.80, 7.00, 8.50, 10.0, 11.5, 13.0]
ZT =   [1.00, 1.15, 1.27, 1.37, 1.43, 1.53, 1.70, 1.88, 2.10, 2.47, 2.88, 3.18, 3.42, 3.60, 3.73, 3.80, 3.84, 3.86]
ZW =   [1.00, 1.08, 1.22, 1.33, 1.42, 1.52, 1.68, 1.80, 1.90, 1.98, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00]
YW =   [0.00, 0.14, 0.27, 0.40, 0.52, 0.62, 0.74, 0.90, 1.08, 1.27, 1.43, 1.54, 1.61, 1.655, 1.675, 1.68, 1.68, 1.68]
ZC =   [1.00, 1.04, 1.13, 1.27, 1.38, 1.43, 1.44, 1.38, 1.27, 1.13, 1.12, 1.22, 1.36, 1.52, 1.62, 1.65, 1.65, 1.65]
YC =   [0.00, 0.18, 0.34, 0.52, 0.66, 0.74, 0.79, 0.92, 1.10, 1.32, 1.50, 1.60, 1.645, 1.665, 1.675, 1.678, 1.678, 1.678]
ZL =   [1.00, 0.92, 0.80, 0.72, 0.68, 0.665, 0.665, 0.61, 0.42, 0.26, 0.21, 0.19, 0.185, 0.18, 0.18, 0.18, 0.18, 0.18]
YL =   [0.00, 0.04, 0.08, 0.13, 0.18, 0.22, 0.30, 0.55, 0.98, 1.20, 1.38, 1.48, 1.53, 1.56, 1.575, 1.585, 1.585, 1.585]
ZB =   [1.00, 0.92, 0.80, 0.72, 0.68, 0.665, 0.665, 0.60, 0.40, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
NEXP = [2.00, 2.00, 2.05, 2.10, 2.15, 2.20, 2.30, 2.40, 2.55, 2.75, 3.00, 3.25, 3.50, 3.75, 3.95, 4.10, 4.20, 4.20]
# how far the crease ledge protrudes (0 = flush).  Bill lip and the side "blade".
LEDGE = [0.00, 0.03, 0.06, 0.09, 0.11, 0.10, 0.06, 0.03, 0.03, 0.06, 0.08, 0.07, 0.04, 0.012, 0.0, 0.0, 0.0, 0.0]
# roundness of the lower segment (1 = quarter ellipse like the bill, 0 = straight skirt)
ROUND = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.75, 0.45, 0.25, 0.12, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

BODY_X = RX[-1]

# point counts per segment (shared end points counted once)
N1, N2, N3, N4 = 9, 13, 11, 41
NSEC = N1 + N2 + N3 + N4 - 3     # points in a half section
CREASE_IDX = N1 + N2 - 2         # index of the crease point
SKIRT_IDX = N1 - 1               # index of lower corner


def rails(X):
    X = np.asarray(X, float)
    f = lambda arr: pchip(RX, arr, X)
    return dict(zt=f(ZT), zw=f(ZW), yw=f(YW), zc=f(ZC), yc=f(YC), zl=f(ZL), yl=f(YL),
                zb=f(ZB), n=f(NEXP), ledge=f(LEDGE), rnd=f(ROUND))


def half_section(X):
    """Return (NSEC, 2) array of (y, z), y >= 0, from bottom centre to top centre."""
    r = rails(X)
    zt, zw, yw, zc, yc, zl, yl, zb, n, ledge, rnd = (float(r[k]) for k in
                                                 ("zt", "zw", "yw", "zc", "yc", "zl", "yl", "zb", "n", "ledge", "rnd"))
    if X <= 1e-6:
        return np.tile([[0.0, S.NOSE_TIP_Z]], (NSEC, 1))
    P = lambda y, z: np.array([y, z])
    # S1 bottom: flat-ish, horizontal tangent at centre
    s1 = bezier(P(0, zb), P(yl * 0.55, zb), P(yl * 0.92, zl + (zb - zl) * 0.25), P(yl, zl), 60)
    # S2 lower side: from lower corner up/outward to the crease
    dz = zc - zl
    a1 = P(yl + (yc - yl) * 0.15, zl + dz * 0.35); a2 = P(yc - 0.02 * dz, zl + dz * 0.80)   # straight-ish skirt
    b1 = P(yl + (yc - yl) * 0.60, zl); b2 = P(yc, zl + dz * 0.45)                          # rounded (bill)
    s2 = bezier(P(yl, zl), a1 * (1 - rnd) + b1 * rnd, a2 * (1 - rnd) + b2 * rnd, P(yc, zc), 80)
    # S3 crease -> max width.  'ledge' pulls the surface just above the crease inwards
    yin = max(min(yc, yw) - ledge, 0.0)
    dz3 = max(zw - zc, 0.02)
    s3 = bezier(P(yc, zc), P(yin, zc + dz3 * 0.12), P(yw, zw - dz3 * 0.45), P(yw, zw), 80)
    # S4 superellipse from (yw, zw) to (0, zt)
    th = np.linspace(0, np.pi / 2, 200)
    c, s = np.cos(th), np.sin(th)
    s4 = np.stack([yw * np.sign(c) * np.abs(c) ** (2 / n), zw + (zt - zw) * np.abs(s) ** (2 / n)], axis=1)
    parts = [resample(s1, N1), resample(s2, N2)[1:], resample(s3, N3)[1:], resample(s4, N4)[1:]]
    return np.concatenate(parts, axis=0)


def stations_end_car():
    """Longitudinal stations for an end car (X from tip), dense on the nose."""
    a = np.concatenate([
        [0.0], np.geomspace(0.004, 0.2, 9),
        np.arange(0.25, 2.0, 0.05), np.arange(2.0, 6.0, 0.10), np.arange(6.0, 13.0, 0.25),
    ])
    b = np.arange(13.0, S.END_CAR_LENGTH - 0.30 + 1e-6, 0.5)
    st = np.unique(np.round(np.concatenate([a, b, [S.END_CAR_LENGTH - 0.30]]), 5))
    return st


def body_section():
    return half_section(BODY_X)


def full_ring(half):
    """Mirror a half section into a closed ring (right side y<0 first, bottom->top, then left top->bottom)."""
    right = half.copy(); right[:, 0] *= -1
    left = half[::-1][1:-1].copy()
    return np.concatenate([right, left], axis=0)


if __name__ == "__main__":
    import sys
    from PIL import Image, ImageDraw
    out = sys.argv[1] if len(sys.argv) > 1 else "shape_check.png"
    W, H, sc = 1800, 900, 60
    im = Image.new("RGB", (W, H), (30, 30, 30)); d = ImageDraw.Draw(im)
    st = stations_end_car()
    # side silhouette (top and bottom) and plan (max half-width)
    top = [(40 + x * sc, 430 - float(rails(x)["zt"]) * sc) for x in st if x < 16]
    bot = [(40 + x * sc, 430 - float(rails(x)["zl"]) * sc) for x in st if x < 16]
    cre = [(40 + x * sc, 430 - float(rails(x)["zc"]) * sc) for x in st if x < 16]
    d.line(top, fill=(255, 255, 255), width=2); d.line(bot, fill=(200, 200, 200), width=2)
    d.line(cre, fill=(255, 120, 0), width=1)
    meas = [(0, 1.0), (1.0, 1.44), (1.25, 1.72), (1.5, 1.84), (2, 2.05), (2.5, 2.30), (3, 2.55), (3.5, 2.75),
            (4, 2.95), (4.5, 3.10), (5, 3.25), (5.5, 3.38), (6, 3.5), (7, 3.62), (8, 3.71), (9, 3.77), (10, 3.81), (11, 3.84), (12, 3.85)]
    for x, z in meas:
        d.ellipse([40 + x * sc - 3, 430 - z * sc - 3, 40 + x * sc + 3, 430 - z * sc + 3], outline=(0, 255, 0))
    plan = [(40 + x * sc, 880 - float(max(rails(x)["yw"], rails(x)["yc"])) * sc * 2) for x in st if x < 16]
    d.line(plan, fill=(120, 200, 255), width=2)
    # sections
    for i, x in enumerate([0.5, 1.0, 1.6, 2.5, 4.0, 6.0, 9.0, 13.0]):
        hs = half_section(x); ring = full_ring(hs)
        ox = 1250 + (i % 4) * 140; oy = 300 + (i // 4) * 300
        d.line([(ox + p[0] * 40, oy - p[1] * 40) for p in np.vstack([ring, ring[:1]])], fill=(255, 255, 0))
        d.text((ox - 20, oy + 10), f"X={x}", fill=(255, 255, 255))
    im.save(out)
    print("NSEC", NSEC, "stations", len(st))
