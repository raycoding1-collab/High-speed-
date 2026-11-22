"""Railway alignment: horizontal clothoid/arc geometry, cant, vertical profile (pure numpy).

Chainage s in metres.  World: x east, y north, z up.  The line starts heading +x.
China runs left-hand on double track: the filmed train (direction +s) uses the LEFT
track, offset +TRACK_CENTRES/2 to the left of the alignment centre line.
"""
import math

import numpy as np

from . import specs as S

S_MIN, S_MAX = -900.0, 5600.0
CURVE = dict(k_start=1200.0, L1=S.TRANSITION_L, arc=1030.0, L2=S.TRANSITION_L, R=S.CURVE_RADIUS, sign=+1)  # left-hand curve
RAIL_Z = 28.0                    # rail level of the line (valley floor ~ 0)
VIADUCT = (-900.0, 3520.0)       # chainage range of the viaduct
CUTTING = (3800.0, 4450.0)
TUNNEL_S = 4450.0                # tunnel portal
TRACK_SIDE = +1                  # left track for the +s train

_ds = 0.25
_s = np.arange(S_MIN, S_MAX + _ds, _ds)


def curvature(s):
    c = CURVE
    a = c["k_start"]; b = a + c["L1"]; d = b + c["arc"]; e = d + c["L2"]
    k0 = c["sign"] / c["R"]
    s = np.asarray(s, float)
    k = np.where(s < a, 0.0,
        np.where(s < b, k0 * (s - a) / c["L1"],
        np.where(s < d, k0,
        np.where(s < e, k0 * (e - s) / c["L2"], 0.0))))
    return k


_k = curvature(_s)
_theta = np.concatenate([[0.0], np.cumsum(0.5 * (_k[1:] + _k[:-1]) * _ds)])
_theta -= np.interp(0.0, _s, _theta)
_x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(_theta[1:]) + np.cos(_theta[:-1])) * _ds)])
_y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(_theta[1:]) + np.sin(_theta[:-1])) * _ds)])
_x -= np.interp(0.0, _s, _x); _y -= np.interp(0.0, _s, _y)


def heading(s):
    return np.interp(s, _s, _theta)


def xy(s):
    return np.interp(s, _s, _x), np.interp(s, _s, _y)


def cant(s):
    """Applied cant (m), proportional to curvature (linear ramps on the clothoids)."""
    return np.abs(curvature(s)) * CURVE["R"] * S.CANT


def bank(s):
    """Roll angle (rad) of the track: positive = left side up.  Left curve -> right (outer) rail up -> negative."""
    return -np.sign(curvature(s)) * np.arcsin(cant(s) / 1.500)


def rail_z(s):
    return np.full_like(np.asarray(s, float), RAIL_Z)


def frame(s, offset=0.0):
    """Track frame at chainage s for a track offset (m, + left) from the centre line.

    Returns origin (3,), tangent T, left N, up U (each (3,) or (n,3)), all at top-of-rail centre of that track,
    including cant (rotation about the track centre line, which is raised by cant/2).
    """
    s = np.asarray(s, float)
    th = heading(s); x, y = xy(s)
    T = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)
    N0 = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)
    Z = np.stack([np.zeros_like(th), np.zeros_like(th), np.ones_like(th)], -1)
    phi = bank(s)[..., None]
    N = N0 * np.cos(phi) + Z * np.sin(phi)
    U = -N0 * np.sin(phi) + Z * np.cos(phi)
    track_c = np.stack([x, y, rail_z(s)], -1) + N0 * offset
    origin = track_c + Z * (cant(s)[..., None] / 2)
    return origin, T, N, U


def track_offset():
    return TRACK_SIDE * S.TRACK_CENTRES / 2


def nearest_s(px, py, s_guess=None):
    """Chainage and signed lateral offset (+left) of a world point (vectorised over points)."""
    px = np.atleast_1d(px); py = np.atleast_1d(py)
    coarse = _s[::40]; cx = _x[::40]; cy = _y[::40]
    d2 = (px[:, None] - cx[None]) ** 2 + (py[:, None] - cy[None]) ** 2
    i = np.argmin(d2, axis=1)
    s0 = coarse[i]
    for _ in range(3):
        th = heading(s0); x0, y0 = xy(s0)
        s0 = s0 + (px - x0) * np.cos(th) + (py - y0) * np.sin(th)
    th = heading(s0); x0, y0 = xy(s0)
    lat = -(px - x0) * np.sin(th) + (py - y0) * np.cos(th)
    return s0, lat


if __name__ == "__main__":
    for s in (0, 1200, 1870, 2400, 2900, 3570, 4450):
        o, T, N, U = frame(s, track_offset())
        print(f"s={s:6.0f}  xy=({o[0]:8.1f},{o[1]:7.1f})  heading={math.degrees(heading(s)):6.2f}deg  "
              f"k={curvature(s)*1e4:5.3f}e-4  cant={cant(s)*1000:5.1f}mm  bank={math.degrees(bank(s)):5.2f}deg")
    v = S.SPEED
    print(f"lateral acc at R: {v*v/S.CURVE_RADIUS:.3f} m/s2, uncompensated {v*v/S.CURVE_RADIUS - 9.81*S.CANT/1.5:.3f} m/s2")
