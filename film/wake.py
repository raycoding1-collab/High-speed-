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
    near = np.hypot(pos[:, 0] - cam[0], pos[:, 1] - cam[1]) < 3.0
    s0[near] += 6.0                                            # nothing spawns right in front of the lens
    th0 = A.heading(s0); cx, cy = A.xy(s0)
    pos = np.stack([cx - np.sin(th0) * lat, cy + np.cos(th0) * lat, np.zeros(n)], -1)
    gz = ground_fn(pos[:, 0], pos[:, 1]) if ground_fn else np.full(n, A.RAIL_Z - 0.85)
    pos[:, 2] = gz + 0.01
    vel = np.zeros((n, 3))
    dt = 1.0 / (SH.FPS * 8)
    n_samples = int((f1 - f0 + 2) * 2)
    target_t = np.array([shot.sim(j / (2.0 * SH.FPS)) for j in range(n_samples)], dtype=np.float64)
    T = float(target_t[-1])
    steps = int(math.ceil(T / dt)) + 1
    out = []
    idx = 0
    while idx < n_samples and target_t[idx] <= 0.0:
        out.append(pos.copy())
        idx += 1
    noise_state = rng.standard_normal((n, 3))
    for k in range(steps):
        t = k * dt
        pos_prev = pos.copy()
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
        t_next = (k + 1) * dt
        while idx < n_samples and target_t[idx] <= t_next + 1e-12:
            alpha = float(np.clip((target_t[idx] - t) / dt, 0.0, 1.0))
            out.append(pos_prev + alpha * (pos - pos_prev))
            idx += 1
        if idx >= n_samples:
            break
    while len(out) < n_samples:
        out.append(pos.copy())
    return np.array(out), kinds


def build_particles(name, coll, traj, kinds, f0, mats, cam=None):
    """Leaves (tumbling 6-14 cm chips) and dust puffs (camera-facing soft billboards that grow from
    invisible specks to 0.6-1.6 m once the slipstream has lifted them), animated by absolute shape keys."""
    rng = np.random.default_rng(1)
    n = traj.shape[1]
    leaf = kinds == 0
    size = np.where(leaf, rng.uniform(0.06, 0.14, n), rng.uniform(0.22, 0.50, n))
    rot = rng.uniform(0, 2 * np.pi, (n, 3))
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    asp = np.where(leaf, 0.55, 1.0)
    p0 = traj[0]

    def quads(P, k):
        disp = np.linalg.norm(P - p0, axis=1)
        grow = np.where(leaf, 1.0, 0.03 + 0.97 * np.clip(disp / 2.5, 0, 1) * (1 + 0.8 * np.clip((disp - 2.5) / 8, 0, 1)))
        sz = size * grow
        V = np.zeros((n * 4, 3))
        if cam is not None:
            d = cam[None, :] - P; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            ux = np.cross(d, np.array([0, 0, 1.0])); ux /= np.linalg.norm(ux, axis=1, keepdims=True) + 1e-9
            uy = np.cross(ux, d)
        for i in range(4):
            cx = corners[i, 0] * sz; cy = corners[i, 1] * sz * asp
            # leaves tumble (spin grows with travelled distance); dust faces the camera
            a = rot[:, 0] + disp * 1.7; b = rot[:, 1] + disp * 1.1
            lx = cx * np.cos(a) - cy * np.sin(a); ly = cx * np.sin(a) + cy * np.cos(a)
            lz = ly * np.sin(b); ly2 = ly * np.cos(b)
            leafv = np.stack([lx, ly2, lz], -1)
            if cam is not None:
                dustv = ux * cx[:, None] + uy * cy[:, None]
            else:
                dustv = leafv
            V[i::4] = P + np.where(leaf[:, None], leafv, dustv)
        return V
    faces = [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(n)]
    uvs = np.tile(np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float), (n, 1))
    ob = new_mesh_object(name, quads(traj[0], 0), faces, coll, uvs=uvs, smooth=False)
    me = ob.data
    me.materials.append(mats["leaf"]); me.materials.append(mats["dust"])
    me.polygons.foreach_set("material_index", kinds.astype(np.int32))
    lid = me.attributes.new("lid", "FLOAT", "FACE")              # stable per-leaf colour (no per-frame flicker)
    lid.data.foreach_set("value", rng.random(n).astype(np.float32))
    ob.shape_key_add(name="Basis", from_mix=False)
    me.shape_keys.use_relative = False
    for k in range(len(traj)):
        sk = ob.shape_key_add(name=f"k{k}", from_mix=False)
        sk.data.foreach_set("co", quads(traj[k], k).astype(np.float32).ravel())
    nkeys = len(traj)
    frames = f0 + np.arange(nkeys) / 2.0
    evals = 10.0 * (np.arange(nkeys) + 1)
    set_keys(me.shape_keys, "eval_time", 0, frames.astype(np.float32), evals.astype(np.float32))
    return ob


def particle_materials():
    leaf = bpy.data.materials.get("WakeLeaf")
    if leaf is None:
        leaf = bpy.data.materials.new("WakeLeaf"); leaf.use_nodes = True
        nt = leaf.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
        # dry leaves and a few green ones, colour fixed per leaf (face attribute 'lid')
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.13, 0.07, 0.025, 1); ramp.color_ramp.elements[1].color = (0.24, 0.14, 0.045, 1)
        ramp.color_ramp.elements.new(0.45).color = (0.19, 0.11, 0.035, 1)
        ramp.color_ramp.elements.new(0.85).color = (0.08, 0.10, 0.03, 1)
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "lid"
        L(at.outputs["Fac"], ramp.inputs[0])
        L(ramp.outputs[0], p.inputs["Base Color"]); p.inputs["Roughness"].default_value = 0.7
        p.inputs["Subsurface Weight"].default_value = 0.15
        # leaf outline instead of a rectangle: |2x|^1.6 + |2y|^2 < 1 on the quad UVs (pointed tips)
        uv = nt.nodes.new("ShaderNodeUVMap"); sp = nt.nodes.new("ShaderNodeSeparateXYZ"); L(uv.outputs[0], sp.inputs[0])
        def term(sock, pw):
            a = nt.nodes.new("ShaderNodeMath"); a.operation = "SUBTRACT"; L(sock, a.inputs[0]); a.inputs[1].default_value = 0.5
            b = nt.nodes.new("ShaderNodeMath"); b.operation = "ABSOLUTE"; L(a.outputs[0], b.inputs[0])
            c = nt.nodes.new("ShaderNodeMath"); c.operation = "MULTIPLY"; L(b.outputs[0], c.inputs[0]); c.inputs[1].default_value = 2.0
            d = nt.nodes.new("ShaderNodeMath"); d.operation = "POWER"; L(c.outputs[0], d.inputs[0]); d.inputs[1].default_value = pw
            return d.outputs[0]
        sm = nt.nodes.new("ShaderNodeMath"); sm.operation = "ADD"; L(term(sp.outputs["X"], 1.6), sm.inputs[0]); L(term(sp.outputs["Y"], 2.0), sm.inputs[1])
        inside = nt.nodes.new("ShaderNodeMath"); inside.operation = "LESS_THAN"; L(sm.outputs[0], inside.inputs[0]); inside.inputs[1].default_value = 1.0
        L(inside.outputs[0], p.inputs["Alpha"])
    dust = bpy.data.materials.get("WakeDust")
    if dust is None:
        dust = bpy.data.materials.new("WakeDust"); dust.use_nodes = True
        nt = dust.node_tree; nt.nodes.clear(); L = nt.links.new
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        uv = nt.nodes.new("ShaderNodeUVMap")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(uv.outputs[0], sep.inputs[0])
        def sub(a):
            n = nt.nodes.new("ShaderNodeMath"); n.operation = "SUBTRACT"; L(a, n.inputs[0]); n.inputs[1].default_value = 0.5
            q = nt.nodes.new("ShaderNodeMath"); q.operation = "MULTIPLY"; L(n.outputs[0], q.inputs[0]); L(n.outputs[0], q.inputs[1])
            return q.outputs[0]
        r2 = nt.nodes.new("ShaderNodeMath"); r2.operation = "ADD"; L(sub(sep.outputs["X"]), r2.inputs[0]); L(sub(sep.outputs["Y"]), r2.inputs[1])
        g = nt.nodes.new("ShaderNodeMath"); g.operation = "MULTIPLY"; g.inputs[1].default_value = -14.0; L(r2.outputs[0], g.inputs[0])
        e = nt.nodes.new("ShaderNodeMath"); e.operation = "EXPONENT"; L(g.outputs[0], e.inputs[0])
        fade = nt.nodes.new("ShaderNodeAttribute"); fade.attribute_type = "OBJECT"; fade.attribute_name = "fade"
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY"; L(e.outputs[0], a.inputs[0]); L(fade.outputs["Fac"], a.inputs[1])
        a2 = nt.nodes.new("ShaderNodeMath"); a2.operation = "MULTIPLY"; a2.inputs[1].default_value = 0.10; L(a.outputs[0], a2.inputs[0])
        df = nt.nodes.new("ShaderNodeBsdfDiffuse"); df.inputs["Color"].default_value = (0.62, 0.56, 0.45, 1)
        tl = nt.nodes.new("ShaderNodeBsdfTranslucent"); tl.inputs["Color"].default_value = (0.62, 0.56, 0.45, 1)
        mx0 = nt.nodes.new("ShaderNodeMixShader"); mx0.inputs[0].default_value = 0.4; L(df.outputs[0], mx0.inputs[1]); L(tl.outputs[0], mx0.inputs[2])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader"); L(a2.outputs[0], mx.inputs[0]); L(tr.outputs[0], mx.inputs[1]); L(mx0.outputs[0], mx.inputs[2])
        L(mx.outputs[0], out.inputs["Surface"])
    return dict(leaf=leaf, dust=dust)


def build_wake_effects(coll, ground_fn):
    """Leaves + dust for the trackside cutting shot (S7) and the tunnel portal (S9)."""
    mats = particle_materials()
    out = []
    for sh, f0, f1 in SH.timeline(SH.build_shots()):
        if sh.name.startswith("S7"):
            traj, kinds = simulate(sh, f0, f1, 900, 800, (60, 6.5), 3, ground_fn)
        elif sh.name.startswith("S9"):
            traj, kinds = simulate(sh, f0, f1, 500, 500, (55, 6.0), 5, ground_fn)
        elif sh.name.startswith("SX"):
            traj, kinds = simulate(sh, f0, f1, 700, 600, (60, 6.5), 7, ground_fn)
        else:
            continue
        ob = build_particles(f"Wake_{sh.name}", coll, traj, kinds, f0, mats, cam=np.asarray(sh.cam_fn(0.0)))
        # dust cloud thins out over ~3.5 s after the tail has passed the camera
        cam = np.asarray(sh.cam_fn(0.0))
        s_cam = float(A.nearest_s(cam[0:1], cam[1:2])[0][0])
        t_tail = (s_cam + S.TRAIN_LENGTH - sh.s0) / S.SPEED          # when the tail passes the camera's chainage
        ft = np.array([f0, f0 + sh.screen(t_tail + 0.3) * SH.FPS, f0 + sh.screen(t_tail + 4.0) * SH.FPS, f1 + 1], np.float32)
        ob["fade"] = 1.0
        set_keys(ob, '["fade"]', 0, ft, np.array([1.0, 1.0, 0.15, 0.1], np.float32))
        # visible only during its own shot
        set_keys(ob, "hide_render", 0, np.array([f0 - 1, f0, f1 + 1], np.float32), np.array([1, 0, 1], np.float32), "CONSTANT")
        out.append(ob)
    return out
