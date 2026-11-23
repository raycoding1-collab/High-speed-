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
    def __init__(self, name, dur, s0, lens, desc, cam_fn, aim_fn=None, attach=None, fstop=None, focus=None, shake=0.0,
                 jolt=False, ramp=None):
        self.name, self.s0, self.lens, self.desc = name, s0, lens, desc
        self.cam_fn, self.aim_fn, self.attach, self.fstop, self.focus, self.shake = cam_fn, aim_fn, attach, fstop, focus, shake
        self.jolt = jolt          # pressure-wave kick of a trackside camera at the nose and tail passes
        # speed ramp (high-speed camera): ramp = (t_sim_start, t_sim_end, playback speed); everything is simulated in
        # real (sim) time, the edit plays the slow segment back slower, with 0.35 s cosine ramps in and out.
        # dur = duration on screen, sim_dur = real time covered by the shot.
        self.sim_dur = dur
        self.ramp = ramp
        u = np.linspace(0.0, dur + 1.0, int((dur + 1.0) * 2000) + 1)
        v = np.ones_like(u)
        if ramp:
            a, b, sp = ramp[:3]
            r_in, r_out = (ramp[3], ramp[4]) if len(ramp) > 3 else (0.35, 0.35)
            w = np.clip((u - (a - r_in)) / r_in, 0, 1) * np.clip(((b + r_out) - u) / r_out, 0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            v = np.exp(np.log(sp) * w)          # speed eases exponentially: a snap into slow motion, not a slur
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (1 / v[1:] + 1 / v[:-1]) * np.diff(u))])
        self._u, self._tau, self._v = u, tau, v
        self.dur = float(np.interp(dur, u, tau))

    def sim(self, tau):
        """Screen time since the cut -> real (simulation) time."""
        return float(np.interp(tau, self._tau, self._u)) if np.isscalar(tau) else np.interp(tau, self._tau, self._u)

    def screen(self, t):
        """Real (simulation) time -> screen time since the cut."""
        return float(np.interp(t, self._u, self._tau)) if np.isscalar(t) else np.interp(t, self._u, self._tau)

    def speed(self, t):
        """Playback speed at real time t (1 = real time, 0.25 = four times slower)."""
        return float(np.interp(t, self._u, self._v))

    def s_head(self, t):
        return self.s0 + V * t


def _smooth_follow(target_fn, t, lag=0.35):
    """Operator follow: aim at where the target was `lag` s ago blended with now (critically damped feel)."""
    return 0.6 * target_fn(max(t - lag, 0)) + 0.4 * target_fn(t)


def pressure_jolt(sh, t, s_cam):
    """Trackside camera hit by the train's pressure waves (radians / metres).

    The nose pulse (a sharp +/- pressure swing lasting ~nose length / V = 0.14 s) kicks the small camera stand,
    which rings at ~7.5 Hz with ~0.22 s decay; the tail pulse comes 211 m / V = 1.9 s later with opposite sign
    and ~60 % strength; behind the tail the turbulent wake buffets the stand for ~1-2 s."""
    t_n = (s_cam - sh.s0) / V
    t_t = t_n + S.TRAIN_LENGTH / V
    def ring(x):
        return 0.0 if x < 0 else math.exp(-x / 0.22) * math.sin(2 * math.pi * 7.5 * x)
    k = ring(t - t_n) - 0.6 * ring(t - t_t)
    w = 0.0
    if t > t_t:
        x = t - t_t
        w = math.exp(-x / 0.9) * (math.sin(2 * math.pi * 3.1 * x + 0.7) + 0.6 * math.sin(2 * math.pi * 5.3 * x + 2.1)
                                  + 0.4 * math.sin(2 * math.pi * 8.9 * x + 4.0)) / 2.0
    yaw = math.radians(0.8) * k + math.radians(0.15) * w
    pitch = math.radians(0.5) * ring(t - t_n - 0.03) - math.radians(0.3) * ring(t - t_t - 0.03) + math.radians(0.1) * w
    lateral = 0.010 * k + 0.003 * w
    return yaw, pitch, lateral


def build_shots():
    shots = []
    # S1 establishing aerial: drone 700 m south of the viaduct, drifting east at 6 m/s
    # (v3) the drone starts low over the hamlet south of the line (roofs, orchards, ponds pass below), flies north
    # at 14 m/s and tilts up from the village to the viaduct as the train crosses the valley
    def _s1_cam(t):
        return _rail(1500 + 2.0 * t, -560 + 14.0 * t, 20 - 0.5 * t)
    def _s1_aim(t, sh=None):
        w = 0.5 - 0.5 * math.cos(math.pi * min(max((t - 1.0) / 4.0, 0.0), 1.0))     # village -> viaduct pan
        village = _rail(1505 + 2.0 * t, -430 + 14.0 * t, -26.0)
        line = _rail(1560 + 9 * t, 40, 6)
        return village * (1 - w) + line * w
    shots.append(Shot("S1_aerial", 8.0, 1040.0, 40,
                      "Drone 45 m over the hamlet south of the line, flying north at 14 m/s and tilting up from the village to the viaduct as the train crosses the valley",
                      _s1_cam, _s1_aim, fstop=8))
    # S2 paddy-level, tripod 45 m south of the deck edge next to a pier, looking up
    c2 = _rail(986, 60, 0); c2[2] = _ground_at(c2[0], c2[1]) + 1.55      # north side = the near (left-running) track
    shots.append(Shot("S2_paddy_low", 5.0, 986 - V * 2.4 + 105, 24,
                      "Tripod in the rice field 60 m north of the viaduct, 1.55 m high, looking up: the train streaks across 26 m above the paddies",
                      lambda t, c=c2: c, lambda t: _rail(1018, 0, -12.0), fstop=5.6))   # v3: tilt down: rice + hamlet
    # S3 drone that cannot keep up: 26 m/s (DJI Inspire 3 max), 24 m south of the line, 12 m above deck
    shots.append(Shot("S3_drone", 7.0, 1650 - 300, 24,
                      "Drone flying alongside at its 26 m/s maximum; the train overtakes at 85 m/s relative and leaves it behind",
                      lambda t: _rail(1650 + 26 * t, -24, 12),
                      None, fstop=4.0))
    # S4 helicopter chase at 70 m/s (AS350 cruise), 160 m off and only 7 m above the deck, 135 mm on a gimbal:
    # a long-lens panning shot whose background is the far side of the valley (hamlets, groves, orchards, the
    # karst towers) streaking past, like real helicopter footage of high-speed trains
    # (v3) it flies ahead of the train on the north side and looks back, so the low morning sun is behind the lens
    # (sun 120-175 deg from the view direction): the nose is front-lit while it closes from 440 m to 160 m
    shots.append(Shot("S4_heli", 9.0, 1560.0, 135,
                      "Helicopter at 70 m/s (252 km/h) ahead of the train just above deck height, looking back: the 400 km/h train closes in from 440 m",
                      lambda t: _rail(1960 + 70 * t, 160, 7),
                      None, fstop=5.6, shake=0.08))
    # S5 cab view
    shots.append(Shot("S5_cab", 8.0, 2955.0, 24,
                      "Driver's eye view through the windscreen (2.55 m above rail): masts every 48 m flash past every 0.43 s",
                      None, None, attach=("Car1", (S.END_FRONT_BOGIE_X + S.TRUCK_CENTRES / 2 - 5.85, 0.0, 2.88)), fstop=4.0, shake=0.004))
    # S6 roof camera looking back at the raised pantograph on car 6
    shots.append(Shot("S6_panto", 5.0, 2250.0, 30,
                      "Roof-mounted camera on car 6 looking back at the raised pantograph; contact point slides +-300 mm",
                      None, None, attach=("Car6", (9.5, -1.15, 4.75)), fstop=8, shake=0.002))
    # S7 low trackside in the cutting. Showstopper: the robotic head whip-pans with the nose (~270 deg/s, from the
    # up-line view to abeam of the bodyside) exactly as the camera snaps into 240 fps; 0.26 s of real time fill 2.6 s
    def _s7_aim(t, c=_rail(4150, 6.4, 0.30)):
        # the pan is an angular move of the head (bounded rate, <= ~340 deg/s at the peak), not a target chase
        d0 = _rail(4150 - 60, A.track_offset(), 1.6) - c; d0 /= np.linalg.norm(d0)
        d1 = _rail(4150 + 12.0, A.track_offset(), 1.7) - c; d1 /= np.linalg.norm(d1)
        w = min(max((t - 2.48) / 0.54, 0.0), 1.0); w = w * w * (3 - 2 * w)
        om = math.acos(float(np.clip(d0 @ d1, -1, 1)))
        d = (math.sin((1 - w) * om) * d0 + math.sin(w * om) * d1) / math.sin(om)
        return c + d * 10.0
    c7 = _rail(4150, 6.4, 0.30)        # low remote camera beside the cess, 0.3 m above rail-head height
    shots.append(Shot("S7_cutting_low", 6.0, 4150 - V * 2.7 + 0.0, 20,
                      "Remote low camera in the cess ditch, 3.0 m from the nearest rail, 2.0 m from the car body; pass-by then wake",
                      lambda t, c=c7: c, _s7_aim, fstop=2.8, jolt=True,
                      ramp=(2.64, 2.90, 0.10, 0.12, 0.18)))   # SHOWSTOPPER 1: 240 fps (10 %) as the nose passes
    # S8 long lens head-on from the hill above the cutting
    c8 = _rail(3905, 72.0, 0); c8[2] = _ground_at(c8[0], c8[1]) + 7.0      # north rim of the cutting (on a camera tower), sun behind the lens
    shots.append(Shot("S8_tele", 9.0, 2650.0, 600,
                      "600 mm from the north rim of the cutting, 0.5-1.3 km away; compressed head-on approach out of the curve",
                      lambda t, c=c8: c, lambda t, sh=None: _rail(2650 + V * t - 40, A.track_offset(), 2.2), fstop=8))
    # S9 tunnel portal
    c9 = _rail(A.TUNNEL_S + 9.0, 4.0, 0); c9[2] = _ground_at(c9[0], c9[1]) + 1.7   # on the hill right above the portal
    # Showstopper: a cable-cam over the portal pushes in (8 m forward, 2.5 m down at up to 9 m/s) and tilts down to
    # the nose as it dives into the mountain beneath the lens, snapping into 200 fps
    p9 = _rail(A.TUNNEL_S + 1.0, 4.0, 0); p9[2] = c9[2] - 2.5
    def _push(t):
        w = min(max((t - 1.55) / 1.15, 0.0), 1.0)
        return w * w * (3 - 2 * w)
    def _s9_cam(t, c=c9, p=p9):
        return c + (p - c) * _push(t)
    def _s9_aim(t):
        base = _rail(4330, A.track_offset(), 1.0)
        s_nose = 4450 - V * 2.6 + V * t
        nose = _rail(min(s_nose - 3.0, A.TUNNEL_S - 2.0), A.track_offset(), 1.8)
        w = min(max((t - 1.9) / 0.75, 0.0), 1.0); w = w * w * (3 - 2 * w)
        return base * (1 - w) + nose * w
    shots.append(Shot("S9_portal", 6.5, 4450 - V * 2.6, 24,
                      "On the hill above the portal looking down the cutting: the train races at the lens and dives into the mountain beneath it; cut 2 s after the tail is in",
                      _s9_cam, _s9_aim, fstop=8,
                      ramp=(2.50, 2.78, 0.12, 0.12, 0.20)))   # SHOWSTOPPER 2: 200 fps (12 %) as the nose dives in
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
        d = V * sh.sim_dur
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
