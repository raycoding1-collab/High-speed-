"""Train slipstream / wake effects: leaves and dust lifted behind the train, grass whipped.

Air model (train frame, following Baker's slipstream measurements for HSTs): near the body the air is
dragged along +s at up to ~0.35 V in the boundary layer, decaying with lateral distance from the body side
(e-folding ~2.5 m), and continues as a turbulent wake behind the tail that decays over ~3 s, with an
upward component near the ground and random gusts.  Particles respond with first-order drag
(leaf tau ~0.25 s, dust tau ~0.04 s), gravity and ground contact.  Trajectories are baked per frame
into absolute shape keys (exact motion blur).
"""
import math

import bpy
import numpy as np

from . import alignment as A
from . import shots as SH
from . import specs as S
from .bl import new_mesh_object, set_keys


def air_velocity(s_rel_head, s_rel_tail, lat_dist, z, t_since_tail, rng_noise):
    """Air velocity (along, lateral, up) in m/s at points; inputs are arrays."""
    V = S.SPEED
    beside = (s_rel_head < 0) & (s_rel_tail > 0)            # alongside the train body
    behind = s_rel_tail <= 0
    decay_lat = np.exp(-np.maximum(lat_dist - 1.7, 0) / 2.5)
    # boundary layer grows along the train: stronger towards the tail
    frac = np.clip(s_rel_tail / S.TRAIN_LENGTH, 0, 1)
    u_bl = V * (0.08 + 0.22 * (1 - frac)) * decay_lat
    u_wake = V * 0.32 * np.exp(-np.maximum(t_since_tail, 0) / 1.1) * np.exp(-np.maximum(lat_dist - 1.7, 0) / 3.5)
    u = np.where(beside, u_bl, np.where(behind, u_wake, 0.0))
    gust = rng_noise * np.where(beside | behind, 0.35 * u + 0.5, 0.2)[:, None]
    up = np.where(behind, 0.12 * u_wake, 0.04 * u) * np.exp(-z / 1.5)
    out_lat = np.where(behind, 0.10 * u_wake, -0.03 * u)     # mild outward push behind the tail
    return u + gust[:, 0], out_lat + gust[:, 1], up + 0.5 * gust[:, 2]


def simulate(shot, f0, f1, n_leaves=500, n_dust=3200, region=(70, 7.5), seed=3, ground_fn=None):
    """Return positions (frames*2, N, 3) for leaves+dust and their kinds for the shot (frame and half-frame)."""
    rng = np.random.default_rng(seed)
    off = A.track_offset()
    cam = np.asarray(shot.cam_fn(0.0))
    s_c, _ = A.nearest_s(np.array([cam[0]]), np.array([cam[1]]))
    s_c = float(s_c[0])
    n = n_leaves + n_dust
    s0 = rng.uniform(s_c - region[0], s_c + region[0] * 0.6, n)
    _, cam_lat = A.nearest_s(np.array([cam[0]]), np.array([cam[1]]))
    cam_side = 1.0 if cam_lat[0] > off else -1.0
    side = np.where(rng.random(n) < 0.7, cam_side, -cam_side)                             # mostly on the camera side
    lat = off + side * rng.uniform(1.9, region[1], n)
    kinds = np.concatenate([np.zeros(n_leaves), np.ones(n_dust)])
    tau = np.where(kinds == 0, rng.uniform(0.18, 0.35, n), rng.uniform(0.025, 0.06, n))
    gfac = np.where(kinds == 0, 0.35, 0.08)                  # effective gravity (leaves flutter, dust floats)
    th0 = A.heading(s0)
    cx, cy = A.xy(s0)
    pos = np.stack([cx - np.sin(th0) * lat, cy + np.cos(th0) * lat, np.zeros(n)], -1)
    gz = ground_fn(pos[:, 0], pos[:, 1]) if ground_fn else np.full(n, A.RAIL_Z - 0.85)
    pos[:, 2] = gz + 0.01
    vel = np.zeros((n, 3))
    dt = 1.0 / (SH.FPS * 8)
    out = []
    T = (f1 - f0 + 2) / SH.FPS
    steps = int(T / dt) + 1
    noise_state = rng.standard_normal((n, 3))
    for k in range(steps):
        t = k * dt
        s_head = shot.s_head(t)
        s_p, lat_p = A.nearest_s(pos[:, 0], pos[:, 1])
        ld = np.abs(lat_p - off)
        rel_head = s_p - s_head
        rel_tail = s_p - (s_head - S.TRAIN_LENGTH)
        t_tail = (s_head - S.TRAIN_LENGTH - s_p) / S.SPEED
        noise_state = 0.92 * noise_state + 0.39 * rng.standard_normal((n, 3))
        ua, ul, uz = air_velocity(rel_head, rel_tail, ld, pos[:, 2] - gz, t_tail, noise_state)
        th = A.heading(s_p)
        T_ = np.stack([np.cos(th), np.sin(th), 0 * th], -1)
        N_ = np.stack([-np.sin(th), np.cos(th), 0 * th], -1) * np.sign(lat_p - off)[:, None]
        air = T_ * ua[:, None] + N_ * ul[:, None] + np.array([0, 0, 1.0]) * uz[:, None]
        acc = (air - vel) / tau[:, None] - np.array([0, 0, 9.81]) * gfac[:, None]
        vel += acc * dt
        pos += vel * dt
        low = pos[:, 2] < gz
        pos[low, 2] = gz[low]; vel[low, 2] = np.maximum(vel[low, 2], 0) * 0.2; vel[low, :2] *= 0.85
        if k % 4 == 0:                                         # every half frame
            out.append(pos.copy())
    return np.array(out), kinds


def build_particles(name, coll, traj, kinds, f0, mats):
    """Mesh of small quads (leaves) and specks (dust) animated by absolute shape keys."""
    rng = np.random.default_rng(1)
    n = traj.shape[1]
    size = np.where(kinds == 0, rng.uniform(0.03, 0.06, n), rng.uniform(0.004, 0.012, n))
    rot = rng.uniform(0, 2 * np.pi, (n, 3))
    corners = np.array([[-1, -0.6, 0], [1, -0.6, 0], [1, 0.6, 0], [-1, 0.6, 0]], float)

    def quads(P, spin):
        V = np.zeros((n * 4, 3))
        for i in range(4):
            c = corners[i] * size[:, None]
            a = rot[:, 0] + spin; b = rot[:, 1] + spin * 0.7
            x = c[:, 0] * np.cos(a) - c[:, 1] * np.sin(a)
            y = c[:, 0] * np.sin(a) + c[:, 1] * np.cos(a)
            z = y * np.sin(b); y = y * np.cos(b)
            V[i::4] = P + np.stack([x, y, z], -1)
        return V
    faces = [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(n)]
    ob = new_mesh_object(name, quads(traj[0], 0.0), faces, coll, smooth=False)
    me = ob.data
    mi = np.repeat(kinds.astype(np.int32), 1)
    me.materials.append(mats["leaf"]); me.materials.append(mats["dust"])
    me.polygons.foreach_set("material_index", mi)
    ob.shape_key_add(name="Basis", from_mix=False)
    me.shape_keys.use_relative = False
    for k in range(len(traj)):
        sk = ob.shape_key_add(name=f"k{k}", from_mix=False)
        speed = np.linalg.norm(traj[k] - traj[max(k - 1, 0)], axis=1) * SH.FPS * 2
        sk.data.foreach_set("co", quads(traj[k], k * 0.5 * np.clip(speed, 0, 12) * 0.15).astype(np.float32).ravel())
    # absolute keys: eval_time 10 per key block; key k sits at frame f0 + k/2
    nkeys = len(traj)
    frames = f0 + np.arange(nkeys) / 2.0
    evals = 10.0 * (np.arange(nkeys) + 1)
    set_keys(me.shape_keys, "eval_time", 0, frames.astype(np.float32), evals.astype(np.float32))
    return ob


def particle_materials():
    leaf = bpy.data.materials.get("WakeLeaf")
    if leaf is None:
        leaf = bpy.data.materials.new("WakeLeaf"); leaf.use_nodes = True
        p = leaf.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (0.20, 0.16, 0.05, 1); p.inputs["Roughness"].default_value = 0.7
        p.inputs["Subsurface Weight"].default_value = 0.2
    dust = bpy.data.materials.get("WakeDust")
    if dust is None:
        dust = bpy.data.materials.new("WakeDust"); dust.use_nodes = True
        p = dust.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (0.42, 0.36, 0.27, 1); p.inputs["Roughness"].default_value = 1.0
        p.inputs["Alpha"].default_value = 0.55
    return dict(leaf=leaf, dust=dust)


def build_wake_effects(coll, ground_fn):
    """Leaves + dust for the trackside cutting shot (S7) and the tunnel portal (S9)."""
    mats = particle_materials()
    out = []
    for sh, f0, f1 in SH.timeline(SH.build_shots()):
        if sh.name.startswith("S7"):
            traj, kinds = simulate(sh, f0, f1, 450, 2600, (60, 6.5), 3, ground_fn)
        elif sh.name.startswith("S9"):
            traj, kinds = simulate(sh, f0, f1, 300, 1800, (45, 6.0), 5, ground_fn)
        else:
            continue
        ob = build_particles(f"Wake_{sh.name}", coll, traj, kinds, f0, mats)
        # visible only during its own shot
        set_keys(ob, "hide_render", 0, np.array([f0 - 1, f0, f1 + 1], np.float32), np.array([1, 0, 1], np.float32), "CONSTANT")
        out.append(ob)
    return out
