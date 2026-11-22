"""Physically driven sound for the film, synthesised from the same kinematics as the picture.

Sources (moving with the train): rolling noise at each bogie (wheel/rail), turbulent boundary layer along
the body, nose/tail aero, pantograph (raised: car 6, lowered: car 3).  Each source is band-limited noise in
8 octave bands with published-shape spectra (aero ~ V^6, rolling ~ V^3), propagated to the camera's
microphone with retarded time (true Doppler), spherical spreading, ISO 9613-1 air absorption per band,
a ground reflection, deck/parapet shielding and tunnel-portal radiation.  Plus wind on the microphone,
wake gusts and a quiet natural ambience.  Master: EBU R128 (-23 LUFS, true peak <= -1 dBTP).

usage: python -m film.audio out.wav [--fs 48000]
"""
import math
import sys

import numpy as np
from scipy import signal

from . import alignment as A
from . import shots as SH
from . import specs as S
from . import train as TR

C_AIR = 343.0
FS = 48000
BANDS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
# ISO 9613-1 attenuation, dB/m at 20 C, 70 % RH (approx.)
AIR_DB_PER_M = np.array([0.0001, 0.0004, 0.0011, 0.0024, 0.0045, 0.0082, 0.022, 0.075])
# source spectra (relative dB per octave band) shaped after published HSR pass-by spectra
SPEC = {
    "rolling": np.array([62, 68, 74, 80, 85, 84, 79, 70], float),
    "tbl":     np.array([74, 77, 77, 75, 71, 66, 60, 52], float),
    "nose":    np.array([80, 82, 80, 76, 72, 67, 61, 53], float),
    "tail":    np.array([82, 82, 78, 73, 68, 62, 56, 48], float),
    "panto":   np.array([62, 67, 73, 79, 82, 80, 75, 67], float),
    "bogie":   np.array([70, 74, 76, 76, 73, 69, 63, 55], float),
}


def band_pool(n, rng, variants=6):
    """Pool of independent band-limited noises per octave band, normalised to unit RMS."""
    pool = np.zeros((len(BANDS), variants, n), np.float32)
    for b, fc in enumerate(BANDS):
        lo, hi = fc / math.sqrt(2), min(fc * math.sqrt(2), FS / 2 * 0.95)
        sos = signal.butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        for v in range(variants):
            x = signal.sosfilt(sos, rng.standard_normal(n + 4000))[4000:]
            pool[b, v] = (x / (np.std(x) + 1e-12)).astype(np.float32)
    return pool


def train_sources(res_layout):
    """List of (kind, along_from_nose_m, height_m, lateral_m, weight_dB)."""
    src = []
    for c in res_layout:
        for along in c["bogies"]:
            src.append(("rolling", along, 0.35, 0.0, 0.0))
            src.append(("bogie", along, 0.6, 0.0, -3.0))
        L = c["length"]
        for k in np.arange(4.0, L, 8.0):
            src.append(("tbl", c["start"] + k, 2.2, 0.0, -2.0))
        if c["index"] == 6:
            src.append(("panto", c["centre"], 5.35, 0.0, 3.0))
        if c["index"] == 3:
            src.append(("panto", c["centre"], 4.35, 0.0, -9.0))     # lowered pantograph still disturbs the flow
    src.append(("nose", 1.0, 1.4, 0.0, 0.0))
    src.append(("tail", S.TRAIN_LENGTH - 1.0, 1.6, 0.0, 0.0))
    return src


def speed_scale(kind, v):
    vr = v / (350 / 3.6)
    return {"rolling": 30, "bogie": 50, "tbl": 60, "nose": 60, "tail": 60, "panto": 60}[kind] * math.log10(vr)


def source_world(s_head, along, h, lat, off):
    O, T, N, U = A.frame(np.atleast_1d(s_head - along), off)
    return O + U * h + N * lat


def mic_track(sh, t):
    """Microphone world position for shot sh at times t (array)."""
    if sh.attach:
        return None
    return np.array([sh.cam_fn(float(x)) for x in t])


def render_shot(sh, pool, rng, layout):
    dur = sh.sim_dur                     # real time; speed ramps are applied afterwards (time_warp)
    n = int(round(dur * FS))
    tk = np.linspace(0, dur, int(dur * 500) + 1)                  # control rate 500 Hz
    off = A.track_offset()
    out = np.zeros(n, np.float64)
    t_audio = np.arange(n) / FS
    srcs = train_sources(layout)
    if sh.attach:
        return interior_or_roof(sh, n, rng, pool)
    mic = mic_track(sh, tk)
    # source-time table (allow up to 3 s of travel time)
    tau = np.linspace(-3.0, dur, int((dur + 3.0) * 500) + 1)
    s_tau = sh.s0 + S.SPEED * tau
    portal = source_world(np.array([A.TUNNEL_S]), 0, 3.0, 0, off)[0]
    deck_top = A.RAIL_Z + 1.15 - 0.72
    for i, (kind, along, h, lat, wdb) in enumerate(srcs):
        P = source_world(s_tau, along, h, lat, off)                 # (len(tau), 3)
        s_src = s_tau - along
        inside = s_src > A.TUNNEL_S + 2
        # solve retarded time t_e for each control time t
        te = tk.copy()
        for _ in range(5):
            Pe = np.stack([np.interp(te, tau, P[:, k]) for k in range(3)], -1)
            # sources inside the tunnel radiate from the portal (with in-tunnel loss)
            ins = np.interp(te, tau, inside.astype(float)) > 0.5
            Pe[ins] = portal
            r = np.linalg.norm(Pe - mic, axis=1)
            te = tk - r / C_AIR
        ins = np.interp(te, tau, inside.astype(float)) > 0.5
        depth = np.maximum(np.interp(te, tau, s_src) - A.TUNNEL_S, 0)
        r = np.maximum(r, 1.0)
        # geometric + shielding
        base = SPEC[kind] + wdb + speed_scale(kind, S.SPEED) - 94.0
        g_spread = 25.0 / r
        # deck / parapet shielding for receivers below the deck looking up at track-level sources
        shield = np.zeros_like(r)
        below = mic[:, 2] < deck_top - 2
        if below.any() and h < 1.5:
            shield = np.where(below, -14.0, 0.0)
        elif below.any() and h < 3.0:
            shield = np.where(below, -7.0, 0.0)
        tun = np.where(ins, -8.0 - 0.25 * depth, 0.0)
        # directivity: rolling noise radiates mostly sideways, nose aero forward
        Pe_dir = Pe - mic
        th = np.interp(te, tau, np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0]))))
        fwd = np.stack([np.cos(th), np.sin(th), 0 * th], -1)
        cosang = np.sum(-Pe_dir * fwd, axis=1) / r                   # +1: mic ahead of the source
        dirdb = {"rolling": -3.0 * cosang ** 2, "nose": 2.0 * cosang, "tail": -2.0 * cosang}.get(kind, 0.0 * cosang)
        # ground reflection (image source), only meaningful for low receivers
        h_mic = mic[:, 2] - np.interp(te, tau, P[:, 2]) + h
        te_s = np.interp(t_audio, tk, te)
        idx_base = (te_s + 3.0) * FS + i * 0.731 * FS
        for b in range(len(BANDS)):
            air = 10 ** (-AIR_DB_PER_M[b] * r / 20)
            gdb = base[b] + shield + tun + dirdb
            g = 10 ** (gdb / 20) * g_spread * air
            gs = np.interp(t_audio, tk, g)
            v = (i * 3 + b) % pool.shape[1]
            x = pool[b, v]
            idx = np.mod(idx_base, len(x) - 2)
            i0 = idx.astype(np.int64); f = idx - i0
            sig = x[i0] * (1 - f) + x[i0 + 1] * f
            out += sig * gs
    # wake gust: turbulent air arriving at the microphone after the train passes close by
    s_mic, lat_mic = A.nearest_s(mic[:, 0], mic[:, 1])
    dist_track = np.abs(lat_mic - off)
    tail_s = sh.s0 + S.SPEED * tk - S.TRAIN_LENGTH
    passed = (tail_s > s_mic) & (dist_track < 15) & (np.abs(mic[:, 2] - A.RAIL_Z) < 6)
    if passed.any():
        t_pass = tk[np.argmax(passed)]
        env = np.where(tk > t_pass, np.exp(-(tk - t_pass) / 1.4) * (1 - np.exp(-(tk - t_pass) / 0.15)), 0.0)
        out += wind_noise(n, rng, 0.9) * np.interp(t_audio, tk, env) * 0.5
    return out


def time_warp(sh, x, n_screen):
    """Play a real-time track back with the shot's speed ramp, like the sound of high-speed-camera footage slowed
    down: sample at screen time tau is the real-time sample at sim(tau), so pitch drops with the playback speed."""
    if not sh.ramp:
        return x[:n_screen] if len(x) >= n_screen else np.pad(x, (0, n_screen - len(x)))
    ts = sh.sim(np.arange(n_screen) / FS) * FS
    return np.interp(ts, np.arange(len(x)), x)


def interior_or_roof(sh, n, rng, pool):
    t = np.arange(n) / FS
    if sh.name.startswith("S5"):
        # cab: structure-borne rumble + muffled aero on the windscreen, no Doppler
        rum = signal.sosfilt(signal.butter(2, [25, 140], "band", fs=FS, output="sos"), rng.standard_normal(n))
        aero = signal.sosfilt(signal.butter(2, [120, 1400], "band", fs=FS, output="sos"), rng.standard_normal(n))
        hiss = signal.sosfilt(signal.butter(2, [1500, 5000], "band", fs=FS, output="sos"), rng.standard_normal(n))
        mod = 1 + 0.15 * np.sin(2 * np.pi * 0.31 * t) + 0.08 * np.sin(2 * np.pi * 1.7 * t + 1)
        # rail joints are welded; slab-track gives a fine-grained rumble modulated at the fastener pitch (111/0.61 Hz)
        x = (0.020 * rum / np.std(rum) + 0.012 * aero / np.std(aero) * mod + 0.0025 * hiss / np.std(hiss)) * 1.0
        x = x + traction_hum(n, rng) * 0.0042 + mast_pulses(sh, n, rng, 0.006, 300.0, 1800.0)
        return x
    # roof camera: violent wind on the microphone + pantograph hiss nearby + a whoosh at every passing mast
    w = wind_noise(n, rng, 1.0) * 0.05
    ph = signal.sosfilt(signal.butter(2, [600, 6000], "band", fs=FS, output="sos"), rng.standard_normal(n))
    return w + 0.010 * ph / np.std(ph) + mast_pulses(sh, n, rng, 0.03, 80.0, 2500.0)


def traction_hum(n, rng):
    """Electric drive heard in the cab at 400 km/h, at a low level under the rumble:
    traction-transformer magnetostriction at 2 x 50 Hz and harmonics, and the motor/inverter: wheel 38.4 rev/s x
    gear ratio 2.79 -> rotor 107 rev/s, 4-pole -> 214 Hz electrical, single-pulse inverter harmonics 6k x f
    (1284, 2568 Hz).  Each line wanders slightly (speed and load fluctuation), amplitude breathes."""
    t = np.arange(n) / FS
    out = np.zeros(n)
    wander = signal.sosfilt(signal.butter(1, 0.5, "low", fs=FS, output="sos"), rng.standard_normal(n))
    wander = wander / (np.std(wander) + 1e-12) * 0.0025
    for f, a in ((100.0, 1.0), (200.0, 0.45), (300.0, 0.2), (214.0, 0.6), (1284.0, 0.22), (2568.0, 0.10)):
        motor = f not in (100.0, 200.0, 300.0)
        ph = 2 * np.pi * np.cumsum(f * (1 + (wander if motor else 0.0002 * wander))) / FS
        am = 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.07, 0.3) * t + rng.uniform(0, 6))
        out += a * am * np.sin(ph + rng.uniform(0, 6))
    return out / (np.std(out) + 1e-12)


def mast_pulses(sh, n, rng, amp, lo, hi):
    """Short pressure whooshes each time the train-mounted microphone passes an OCS mast (every 48 m:
    111.1 / 48 = 2.31 per second), filtered broadband noise with a fast attack and ~60 ms decay."""
    t = np.arange(n) / FS
    period = S.MAST_SPACING_VIADUCT / S.SPEED
    phase0 = rng.uniform(0, period)
    k = np.floor((t - phase0) / period)
    x = t - phase0 - k * period
    env = np.where(x >= 0, (1 - np.exp(-x / 0.008)) * np.exp(-x / 0.06), 0.0)
    nz = signal.sosfilt(signal.butter(2, [lo, hi], "band", fs=FS, output="sos"), rng.standard_normal(n))
    return amp * env * nz / (np.std(nz) + 1e-12)


def near_track_layers(sh, n, rng, peak_rms):
    """For microphones close to the line (S7 in the cess, S9 above the portal):
    - singing rails: the approaching wheels excite bending waves that run ahead along the rails and radiate
      near the microphone; on slab track with soft pads they decay ~0.1 dB per metre above 1 kHz, so the metallic hiss (pinned-pinned
      resonance ~1 kHz at 0.61 m fastener pitch, plus harmonics and broadband 1-5 kHz) rises out of the
      silence ~2 s before the nose arrives and is masked once the train is there;
    - the pressure pulses of nose and tail (quasi-static +/- swing lasting ~nose length / V = 0.14 s) that
      hit the microphone as a deep thump (only the 20-60 Hz tail of the pulse survives the mic's high-pass)."""
    t = np.arange(n) / FS
    off = A.track_offset()
    p0 = np.asarray(sh.cam_fn(0.0), float)
    s_mic, lat = A.nearest_s(np.array([p0[0]]), np.array([p0[1]]))
    s_mic = float(s_mic[0]); dist = math.hypot(float(lat[0]) - off, p0[2] - A.RAIL_Z)
    if dist > 25.0:
        return np.zeros(n)
    near = 3.0 / max(dist, 3.0)                                  # line-source spreading from the rail to the mic
    d_nose = s_mic - (sh.s0 + S.SPEED * t)                       # metres of rail between the nose and the mic
    lvl = np.where(d_nose > 0, 10 ** (-0.10 * d_nose / 20), 1.0)
    lvl = np.where(d_nose < -S.TRAIN_LENGTH, np.exp((d_nose + S.TRAIN_LENGTH) / 15.0), lvl)     # ring-down
    wn = rng.standard_normal(n)
    sing = np.zeros(n)
    for fc, q, a in ((960.0, 9.0, 1.0), (1930.0, 10.0, 0.6), (3150.0, 12.0, 0.35)):
        b, a_ = signal.iirpeak(fc, q, fs=FS)
        y = signal.lfilter(b, a_, wn); sing += a * y / (np.std(y) + 1e-12)
    bb = signal.sosfilt(signal.butter(2, [1000, 5000], "band", fs=FS, output="sos"), wn)
    sing += 0.5 * bb / (np.std(bb) + 1e-12)
    sing = sing / (np.std(sing) + 1e-12) * lvl * 10 ** (-10 / 20) * near
    # pressure pulses: derivative-of-Gaussian at the nose, opposite sign at the tail
    t_n = (s_mic - sh.s0) / S.SPEED + dist / C_AIR
    t_t = t_n + S.TRAIN_LENGTH / S.SPEED
    def pulse(tc, sigma=0.045):
        u = (t - tc) / sigma
        return -u * np.exp(-0.5 * u * u)
    pr = pulse(t_n) - 0.6 * pulse(t_t)
    pr = signal.sosfilt(signal.butter(2, 18, "high", fs=FS, output="sos"), pr)
    pr = signal.sosfilt(signal.butter(2, 90, "low", fs=FS, output="sos"), pr)
    pr = pr / (np.max(np.abs(pr)) + 1e-12) * 10 ** (-2 / 20) * near
    return (sing + pr) * peak_rms


def wind_noise(n, rng, strength=1.0):
    """Low-frequency microphone wind rumble with gusts (filtered noise with random AM)."""
    x = signal.sosfilt(signal.butter(2, [20, 380], "band", fs=FS, output="sos"), rng.standard_normal(n))
    x /= np.std(x) + 1e-12
    gust = signal.sosfilt(signal.butter(1, 0.6, "low", fs=FS, output="sos"), rng.standard_normal(n))
    gust = np.clip(0.6 + 2.5 * gust / (np.std(gust) + 1e-12), 0, None)
    return x * gust * strength


def ambience(n, rng, kind="valley"):
    """Quiet natural bed: breeze in leaves + soft insect texture (no tones)."""
    t = np.arange(n) / FS
    leaves = signal.sosfilt(signal.butter(2, [300, 3500], "band", fs=FS, output="sos"), rng.standard_normal(n))
    leaves /= np.std(leaves)
    sway = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 0.037 * t + 1)
    ins = signal.sosfilt(signal.butter(3, [4200, 7800], "band", fs=FS, output="sos"), rng.standard_normal(n))
    ins /= np.std(ins)
    am = signal.sosfilt(signal.butter(1, 25, "low", fs=FS, output="sos"), np.abs(rng.standard_normal(n)))
    am = am / (np.max(am) + 1e-9)
    return 0.0009 * leaves * sway + (0.00035 if kind == "valley" else 0.00015) * ins * am


def master(x, target_lufs=-23.0, tp_db=-1.0):
    import pyloudnorm as pyln
    # gentle high shelf above 9 kHz (nothing harsh) and DC/rumble cut below 20 Hz
    x = signal.sosfilt(signal.butter(2, 20, "high", fs=FS, output="sos"), x, axis=0)
    x = signal.sosfilt(signal.butter(1, 9000, "low", fs=FS, output="sos"), x, axis=0) * 0.35 + x * 0.65
    # slow compressor so that the loudest pass-bys stay comfortable (ratio 2.5 above -18 dBFS RMS)
    env = np.sqrt(signal.sosfilt(signal.butter(1, 4.0, "low", fs=FS, output="sos"), np.mean(x ** 2, axis=1)) + 1e-12)
    ref = np.percentile(env, 99.5)
    thr = ref * 10 ** (-10 / 20)
    gain = np.where(env > thr, (env / thr) ** (1 / 2.5 - 1), 1.0)
    x = x * gain[:, None]
    meter = pyln.Meter(FS)
    lufs = meter.integrated_loudness(x)
    x = x * 10 ** ((target_lufs - lufs) / 20)
    # true-peak limiter (4x oversampled detection, 5 ms look-ahead, smooth release)
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.max(np.abs(up).reshape(-1, 4, x.shape[1]), axis=(1, 2))
    lim = 10 ** (tp_db / 20) * 0.98
    g = np.minimum(1.0, lim / np.maximum(pk, 1e-9))
    la = int(0.005 * FS)
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(g, size=2 * la + 1)
    g = signal.sosfilt(signal.butter(1, 20, "low", fs=FS, output="sos"), g)
    g = np.minimum(g, minimum_filter1d(np.minimum(1.0, lim / np.maximum(pk, 1e-9)), 3))
    x = x * g[:, None]
    final_lufs = meter.integrated_loudness(x)
    up = signal.resample_poly(x, 4, 1, axis=0)
    tp = 20 * np.log10(np.max(np.abs(up)) + 1e-12)
    return x, final_lufs, tp


# per-shot fader: peak 400 ms RMS of the train component (dBFS), like a re-recording mixer would set it
SHOT_PEAK_DB = {"S1": -34, "S2": -21, "S3": -19, "S4": -25, "S5": -27, "S6": -22, "S7": -13, "S8": -33, "S9": -18}
AMB_DB = -41.0


def _peak_rms_db(x, win=0.4):
    k = int(win * FS)
    p = np.convolve(x ** 2, np.ones(k) / k, mode="valid")
    return 10 * np.log10(np.max(p) + 1e-20)


def _rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def render(out_path, seed=5):
    import soundfile as sf
    rng = np.random.default_rng(seed)
    shots = SH.build_shots()
    layout = TR.car_layout()
    pool = band_pool(FS * 20, rng)
    tracks = []
    for sh, f0, f1 in SH.timeline(shots):
        n = int(round(sh.dur * FS))                       # screen samples
        n_sim = int(round(sh.sim_dur * FS))               # real-time samples
        tr = render_shot(sh, pool, rng, layout)
        tr = tr * 10 ** ((SHOT_PEAK_DB[sh.name[:2]] - _peak_rms_db(tr)) / 20)
        if not sh.attach:
            tr = tr + near_track_layers(sh, n_sim, rng, 10 ** (SHOT_PEAK_DB[sh.name[:2]] / 20))
        tr = time_warp(sh, tr, n)
        amb = ambience(n, rng, "valley" if sh.name[:2] in ("S1", "S2", "S3", "S4") else "hill")
        amb = amb * 10 ** ((AMB_DB - _rms_db(amb)) / 20)
        if not sh.attach:
            wn = wind_noise(n, rng, 1.0)
            wind_db = -33.0 if sh.name[:2] in ("S3", "S4") else -47.0
            tr = tr + amb + wn * 10 ** ((wind_db - _rms_db(wn)) / 20)
        if sh.name.startswith("S9"):
            fade = np.clip((sh.dur - np.arange(n) / FS) / 2.5, 0, 1)
            tr = tr * fade
        # stereo: simple panning from source bearing is overkill here; decorrelated wide ambience + centred train
        amb2 = ambience(n, rng) if not sh.attach else 0.0
        if not sh.attach:
            amb2 = amb2 * 10 ** ((AMB_DB - 3 - _rms_db(amb2)) / 20)
        st = np.stack([tr + 0.3 * amb2, tr - 0.3 * amb2], -1) if not sh.attach else np.stack([tr, tr], -1)
        tracks.append(st)
        print(f"[audio] {sh.name}: {sh.dur:.1f} s", flush=True)
    # 20 ms crossfades at cuts
    xf = int(0.02 * FS)
    w = np.linspace(0, 1, xf)[:, None]
    for k, tr in enumerate(tracks):
        if k > 0:
            tr[:xf] *= w
        if k < len(tracks) - 1:
            tr[-xf:] *= w[::-1]
    mix = np.concatenate(tracks)
    x, lufs, tp = master(mix)
    sf.write(out_path, x.astype(np.float32), FS, subtype="PCM_24")
    print(f"[audio] wrote {out_path}: {len(x)/FS:.2f} s, integrated {lufs:.2f} LUFS, true peak {tp:.2f} dBTP")
    return lufs, tp


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "film_audio.wav")
