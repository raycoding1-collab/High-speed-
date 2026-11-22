"""Shot list: every camera move and the train's chainage per frame, with physics checks.

The train runs at a constant 400 km/h (111.11 m/s) on the left track.  s_head = chainage of car 1's nose tip.
Each shot has its own head chainage at t=0 (a cut lets us re-place the train).  Cameras are defined in
world space (or relative to the train) with real-world speeds.
"""
import math

import numpy as np

from . import alignment as A
from . import specs as S

FPS = 24
V = S.SPEED


def _rail(s, lat=0.0, dz=0.0):
    O, T, N, U = A.frame(np.array([s]), 0.0)
    th = A.heading(s)
    n0 = np.array([-math.sin(th), math.cos(th), 0.0])
    p = O[0] + n0 * lat + np.array([0, 0, dz])
    return p


def _ground_at(x, y):
    from .world import ground
    return float(ground(np.array([x]), np.array([y]))[0])


class Shot:
    def __init__(self, name, dur, s0, lens, desc, cam_fn, aim_fn=None, attach=None, fstop=None, focus=None, shake=0.0):
        self.name, self.dur, self.s0, self.lens, self.desc = name, dur, s0, lens, desc
        self.cam_fn, self.aim_fn, self.attach, self.fstop, self.focus, self.shake = cam_fn, aim_fn, attach, fstop, focus, shake

    def s_head(self, t):
        return self.s0 + V * t


def _smooth_follow(target_fn, t, lag=0.35):
    """Operator follow: aim at where the target was `lag` s ago blended with now (critically damped feel)."""
    return 0.6 * target_fn(max(t - lag, 0)) + 0.4 * target_fn(t)


def build_shots():
    shots = []
    # S1 establishing aerial: drone 700 m south of the viaduct, drifting east at 6 m/s
    c1 = _rail(1500, -720, 70)
    shots.append(Shot("S1_aerial", 7.0, 1040.0, 40,
                      "Drone 700 m south, 100 m up, drifting east at 6 m/s; train crosses the valley on the viaduct",
                      lambda t, c=c1: c + np.array([6.0 * t, 0, -0.6 * t]),
                      lambda t, sh=None: _rail(1560 + 9 * t, 40, 20), fstop=8))
    # S2 paddy-level, tripod 45 m south of the deck edge next to a pier, looking up
    c2 = _rail(986, 60, 0); c2[2] = _ground_at(c2[0], c2[1]) + 1.55      # north side = the near (left-running) track
    shots.append(Shot("S2_paddy_low", 5.0, 986 - V * 2.4 + 105, 24,
                      "Tripod in the rice field 60 m north of the viaduct, 1.55 m high, looking up: the train streaks across 26 m above the paddies",
                      lambda t, c=c2: c, lambda t: _rail(1018, 0, -6.0), fstop=5.6))
    # S3 drone that cannot keep up: 26 m/s (DJI Inspire 3 max), 24 m south of the line, 12 m above deck
    shots.append(Shot("S3_drone", 7.0, 1650 - 300, 24,
                      "Drone flying alongside at its 26 m/s maximum; the train overtakes at 85 m/s relative and leaves it behind",
                      lambda t: _rail(1650 + 26 * t, -24, 12),
                      None, fstop=4.0))
    # S4 helicopter chase at 70 m/s (AS350 cruise), 180 m off, 60 m above the deck, 135 mm on a gimbal
    shots.append(Shot("S4_heli", 8.0, 2080.0, 135,
                      "Helicopter at 70 m/s (252 km/h) paralleling the curve; the 400 km/h train slowly pulls away",
                      lambda t: _rail(2080 - 120 + 70 * t, -185, 62),
                      None, fstop=5.6, shake=0.08))
    # S5 cab view
    shots.append(Shot("S5_cab", 8.0, 2955.0, 24,
                      "Driver's eye view through the windscreen (2.55 m above rail): masts every 48 m flash past every 0.43 s",
                      None, None, attach=("Car1", (S.END_FRONT_BOGIE_X + S.TRUCK_CENTRES / 2 - 5.85, 0.0, 2.88)), fstop=4.0, shake=0.004))
    # S6 roof camera looking back at the raised pantograph on car 6
    shots.append(Shot("S6_panto", 5.0, 2250.0, 30,
                      "Roof-mounted camera on car 6 looking back at the raised pantograph; contact point slides +-300 mm",
                      None, None, attach=("Car6", (9.5, -1.15, 4.75)), fstop=8, shake=0.002))
    # S7 low trackside in the cutting
    c7 = _rail(4150, 6.4, 0.30)        # low remote camera beside the cess, 0.3 m above rail-head height
    shots.append(Shot("S7_cutting_low", 6.0, 4150 - V * 2.7 + 0.0, 20,
                      "Remote low camera in the cess ditch, 3.0 m from the nearest rail, 2.0 m from the car body; pass-by then wake",
                      lambda t, c=c7: c,
                      lambda t: _rail(4150 - 60, A.track_offset(), 1.6), fstop=2.8))
    # S8 long lens head-on from the hill above the cutting
    c8 = _rail(3905, 72.0, 0); c8[2] = _ground_at(c8[0], c8[1]) + 7.0      # north rim of the cutting (on a camera tower), sun behind the lens
    shots.append(Shot("S8_tele", 7.0, 2650.0, 600,
                      "600 mm from the north rim of the cutting, 0.5-1.3 km away; compressed head-on approach out of the curve",
                      lambda t, c=c8: c, lambda t, sh=None: _rail(2650 + V * t - 40, A.track_offset(), 2.2), fstop=8))
    # S9 tunnel portal
    c9 = _rail(A.TUNNEL_S + 9.0, 4.0, 0); c9[2] = _ground_at(c9[0], c9[1]) + 1.7   # on the hill right above the portal
    shots.append(Shot("S9_portal", 12.0, 4450 - V * 2.6, 24,
                      "On the hill above the portal looking down the cutting: the train races at the lens and dives into the mountain beneath it; dust and leaves settle",
                      lambda t, c=c9: c, lambda t: _rail(4330, A.track_offset(), 1.0), fstop=8))
    return shots


def timeline(shots):
    out = []
    f = 1
    for sh in shots:
        n = int(round(sh.dur * FPS))
        out.append((sh, f, f + n - 1))
        f += n
    return out


def physics_table(shots):
    rows = []
    for sh, f0, f1 in timeline(shots):
        n = f1 - f0 + 1
        d = V * sh.dur
        rows.append(dict(shot=sh.name, frames=f"{f0}-{f1}", seconds=sh.dur, speed_kmh=S.SERVICE_SPEED_KMH,
                         speed_ms=round(V, 3), wheel_rps=round(S.WHEEL_REV_PER_S, 2),
                         wheel_deg_per_frame=round(360 * S.WHEEL_REV_PER_S / FPS, 1),
                         distance_m=round(d, 1), s_head=f"{sh.s0:.0f}->{sh.s0 + d:.0f}",
                         passby_s=round(S.TRAIN_LENGTH / V, 2), lens_mm=sh.lens, desc=sh.desc))
    return rows


if __name__ == "__main__":
    sh = build_shots()
    tot = sum(s.dur for s in sh)
    print(f"total {tot:.1f} s, {int(tot*FPS)} frames")
    for r in physics_table(sh):
        print(r)
