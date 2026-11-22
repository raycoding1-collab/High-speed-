"""Rural worldbuilding around the hamlets, placed where the cameras look (v3).

Everything here is real-scale and typical of the Guilin / Liuzhou countryside:
- 10 kV distribution lines along the village roads: 9 m concrete poles every ~45 m, steel cross-arm, three
  conductors hanging in catenaries (0.5-0.7 m sag);
- fish ponds next to the hamlets (35 x 22 m, earth dikes, still green water);
- rice-straw stacks drying in the harvested fields round the hamlets;
- plastic-film vegetable tunnels (8 m span, 3.4 m high, 40 m long) in a few fields;
- citrus (kumquat / orange) orchards in rows of small dark glossy trees;
- family tombs with white concrete fronts on the slopes above the cutting (seen from the portal and the cab).
KEEPOUT collects (x, y, radius) discs that the vegetation pass keeps clear.
"""
import math

import bpy
import numpy as np

from . import alignment as A
from . import materials as M
from . import world as W
from .bl import new_mesh_object
from .village import MB

KEEPOUT = []


def _mat_straw():
    m = bpy.data.materials.get("PropStraw")
    if m:
        return m
    m = bpy.data.materials.new("PropStraw"); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (30.0, 30.0, 4.0)
    L(tc.outputs["Object"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.0; nz.inputs["Detail"].default_value = 8.0
    L(mp.outputs[0], nz.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (0.16, 0.12, 0.05, 1); cr.color_ramp.elements[1].color = (0.42, 0.33, 0.15, 1)
    L(nz.outputs["Fac"], cr.inputs[0]); L(cr.outputs[0], p.inputs["Base Color"])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Distance"].default_value = 0.03
    L(nz.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.9
    return m


def _place_xy(s, lat):
    th = A.heading(s); x0, y0 = A.xy(s)
    return x0 - np.sin(th) * lat, y0 + np.cos(th) * lat


def _free(x, y, r):
    """Clear of houses, other props, roads, the river, towers and the line."""
    from . import village as V
    for (hx, hy, *_rest) in V.SITES:
        if math.hypot(x - hx, y - hy) < r + 10.0:
            return False
    for (kx, ky, kr) in KEEPOUT:
        if math.hypot(x - kx, y - ky) < r + kr:
            return False
    for (xs, ys) in W.ROADS:
        if np.min(np.hypot(xs - x, ys - y)) < r + 6.0:
            return False
    if not W._clear_of_features(np.array([x]), np.array([y]), r, river=24.0 + r)[0]:
        return False
    s, lat = A.nearest_s(np.array([x]), np.array([y]))
    return abs(float(lat[0])) > r + 30.0


def power_lines(coll):
    con = M.principled("PolesConcrete", (0.42, 0.41, 0.39), 0.0, 0.8)
    steel = M.principled("PoleSteel", (0.30, 0.30, 0.31), 1.0, 0.5)
    ins = M.principled("PoleInsulator", (0.55, 0.56, 0.55), 0.0, 0.3, coat=0.5)
    wire = M.principled("PowerWire", (0.08, 0.08, 0.085), 1.0, 0.45)
    mb = MB()
    n_poles = 0
    for (xs, ys) in W.ROADS:
        d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        st = np.arange(10.0, d[-1] - 5, 45.0)
        if len(st) < 2:
            continue
        px = np.interp(st, d, xs); py = np.interp(st, d, ys)
        gx = np.gradient(px); gy = np.gradient(py); gn = np.hypot(gx, gy) + 1e-9
        ox, oy = gy / gn * 4.2, -gx / gn * 4.2                      # opposite side to the irrigation ditch
        px, py = px + ox, py + oy
        s_, lat_ = A.nearest_s(px, py)
        ok = np.abs(lat_) > 20.0
        g = W.natural_ground(px, py)
        tops = []
        for i in range(len(px)):
            if not ok[i]:
                tops.append(None); continue
            b = (px[i], py[i], g[i] - 1.5); t = (px[i], py[i], g[i] + 9.0)
            mb.cyl(b, t, 0.13, con, segs=8)
            ax, ay = -gy[i] / gn[i], gx[i] / gn[i]                  # cross-arm square to the line of the wires
            ca = (px[i] - ax * 0.85, py[i] - ay * 0.85, g[i] + 8.4); cb = (px[i] + ax * 0.85, py[i] + ay * 0.85, g[i] + 8.4)
            mb.cyl(ca, cb, 0.05, steel, segs=6)
            pts = []
            for f in (-0.75, 0.0, 0.75):
                ix, iy = px[i] + ax * f, py[i] + ay * f
                zt = g[i] + (9.15 if f == 0.0 else 8.62)
                mb.cyl((ix, iy, zt - 0.17), (ix, iy, zt), 0.04, ins, segs=6)
                pts.append((ix, iy, zt))
            tops.append(pts)
            n_poles += 1
        for i in range(len(tops) - 1):
            if tops[i] is None or tops[i + 1] is None:
                continue
            for k in range(3):
                p0 = np.array(tops[i][k]); p1 = np.array(tops[i + 1][k])
                L_ = np.linalg.norm(p1 - p0); sag = 0.012 * L_ + 0.05
                prev = p0
                for t in np.linspace(0, 1, 9)[1:]:
                    q = p0 + (p1 - p0) * t; q[2] -= 4 * sag * t * (1 - t)
                    mb.cyl(prev, q, 0.009, wire, segs=4, caps=False)
                    prev = q
    mats = [con, steel, ins, wire]
    idx = {m.name: i for i, m in enumerate(mats)}
    mb.m = [idx[m.name] for m in mb.m]
    me = mb.make("PowerLines", mats, recalc=False)
    ob = bpy.data.objects.new("PowerLines", me); coll.objects.link(ob)
    return ob, n_poles


def fish_ponds(coll, hamlets, rng):
    water = bpy.data.materials.get("RiverWater")
    dike = M.principled("PondDike", (0.085, 0.10, 0.045), 0.0, 0.9)
    obs = []
    for (hs, hl, hx, hy) in hamlets:
        for k in range(int(rng.integers(1, 4))):
            for attempt in range(25):
                a = rng.uniform(0, 2 * math.pi); r = rng.uniform(95, 170)
                cx, cy = hx + r * math.cos(a), hy + r * math.sin(a)
                if _free(cx, cy, 24.0):
                    break
            else:
                continue
            yaw = A.heading(hs) + rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.1)
            w, d = rng.uniform(28, 42), rng.uniform(18, 26)
            g = float(W.natural_ground(np.array([cx]), np.array([cy]))[0])
            mb = MB()
            mb.box(-w / 2, w / 2, -d / 2, d / 2, g - 1.2, g - 0.30, water, skip=("bottom", "s0", "s1", "s2", "s3"))
            t = 1.6
            for (x0, x1, y0, y1) in ((-w / 2 - t, w / 2 + t, -d / 2 - t, -d / 2), (-w / 2 - t, w / 2 + t, d / 2, d / 2 + t),
                                     (-w / 2 - t, -w / 2, -d / 2, d / 2), (w / 2, w / 2 + t, -d / 2, d / 2)):
                mb.box(x0, x1, y0, y1, g - 1.0, g + 0.35, dike)
            mats = [water, dike]; idx = {m.name: i for i, m in enumerate(mats)}
            mb.m = [idx[m.name] for m in mb.m]
            me = mb.make(f"FishPond{len(obs)}", mats)
            ob = bpy.data.objects.new(me.name, me); coll.objects.link(ob)
            ob.location = (cx, cy, 0.0); ob.rotation_euler = (0, 0, yaw)
            KEEPOUT.append((cx, cy, math.hypot(w, d) / 2 + 2))
            obs.append(ob)
    return obs


def straw_stacks(coll, src, hamlets, rng):
    straw = _mat_straw()
    import bmesh
    srcs = []
    for v in range(3):
        bm = bmesh.new()
        h0 = 1.0 + 0.25 * v
        c = bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=0.85, radius2=0.80, depth=h0)
        for q in c["verts"]:
            q.co.z += h0 / 2
        c2 = bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=0.88, radius2=0.05, depth=0.95)
        for q in c2["verts"]:
            q.co.z += h0 + 0.47
        for q in bm.verts:
            q.co.x += 0.04 * math.sin(7 * q.co.z + v); q.co.y += 0.04 * math.cos(5 * q.co.z + 2 * v)
        me = bpy.data.meshes.new(f"StrawStack{v}"); bm.to_mesh(me); bm.free()
        me.materials.append(straw)
        ob = bpy.data.objects.new(f"StrawStack{v}", me); src.objects.link(ob); ob.hide_render = True
        srcs.append(ob)
    pts = []
    for (hs, hl, hx, hy) in hamlets:
        for g in range(int(rng.integers(2, 5))):
            for attempt in range(20):
                a = rng.uniform(0, 2 * math.pi); r = rng.uniform(70, 230)
                cx, cy = hx + r * math.cos(a), hy + r * math.sin(a)
                if _free(cx, cy, 12.0):
                    break
            else:
                continue
            m = int(rng.integers(4, 12))
            xs = cx + rng.normal(0, 7, m); ys = cy + rng.normal(0, 5, m)
            pts.append(np.column_stack([xs, ys, W.natural_ground(xs, ys) - 0.05]))
            KEEPOUT.append((cx, cy, 14.0))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    W.scatter_tree("PropStrawStacks", coll, P, srcs, (0.85, 1.2), 31, wind=0.0)
    return len(P)


def greenhouses(coll, hamlets, rng, n_sites=6):
    film = M.principled("GreenhouseFilm", (0.62, 0.64, 0.62), 0.0, 0.28, coat=0.6)
    frame = M.principled("GreenhouseFrame", (0.45, 0.46, 0.47), 1.0, 0.4)
    obs = []
    order = rng.permutation(len(hamlets))
    for hi in order[:n_sites]:
        hs, hl, hx, hy = hamlets[hi]
        for attempt in range(30):
            a = rng.uniform(0, 2 * math.pi); r = rng.uniform(120, 260)
            cx, cy = hx + r * math.cos(a), hy + r * math.sin(a)
            if _free(cx, cy, 30.0):
                break
        else:
            continue
        yaw = A.heading(hs) + rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.05)
        ntun = int(rng.integers(2, 6)); L_ = rng.uniform(32, 48)
        g = float(W.natural_ground(np.array([cx]), np.array([cy]))[0])
        mb = MB()
        R = 4.0
        for k in range(ntun):
            x0 = (k - (ntun - 1) / 2) * 8.6
            ang = np.linspace(0, math.pi, 13)
            ys = np.array([-L_ / 2, L_ / 2])
            for j in range(12):
                a0, a1 = ang[j], ang[j + 1]
                p = [(x0 + R * math.cos(a0), ys[0], g + 3.4 * math.sin(a0)), (x0 + R * math.cos(a1), ys[0], g + 3.4 * math.sin(a1)),
                     (x0 + R * math.cos(a1), ys[1], g + 3.4 * math.sin(a1)), (x0 + R * math.cos(a0), ys[1], g + 3.4 * math.sin(a0))]
                mb.poly(p, film)
            for yy in ys:                                            # end walls (film) and hoops (frame)
                mb.poly([(x0 + R * math.cos(a), yy, g + 3.4 * math.sin(a)) for a in ang], film)
            for yy in np.arange(-L_ / 2, L_ / 2 + 0.1, 2.0):
                prev = None
                for a in ang:
                    q = (x0 + (R + 0.03) * math.cos(a), yy, g + 3.43 * math.sin(a))
                    if prev is not None:
                        mb.cyl(prev, q, 0.02, frame, segs=4, caps=False)
                    prev = q
        mats = [film, frame]; idx = {m.name: i for i, m in enumerate(mats)}
        mb.m = [idx[m.name] for m in mb.m]
        me = mb.make(f"Greenhouse{len(obs)}", mats, recalc=False)
        ob = bpy.data.objects.new(me.name, me); coll.objects.link(ob)
        ob.location = (cx, cy, 0.0); ob.rotation_euler = (0, 0, yaw)
        KEEPOUT.append((cx, cy, math.hypot(ntun * 8.6, L_) / 2 + 3))
        obs.append(ob)
    return obs


def orchards(coll, src, hamlets, rng, n_sites=9):
    """Rows of small citrus trees (crown proxies at 1.1-1.5 m crown radius, dark glossy leaves with fruit dots)."""
    leaf = bpy.data.materials.get("CitrusLeaves")
    if leaf is None:
        leaf = W._leaf_material("CitrusLeaves", (0.012, 0.035, 0.010), (0.05, 0.11, 0.025), 14.0, 0.12, 0.10)
        nt = leaf.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
        # oranges: sparse small dots (Voronoi cells < radius) mixed into the base colour
        tc = nt.nodes.new("ShaderNodeTexCoord")
        vo = nt.nodes.new("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = 6.0; L(tc.outputs["Object"], vo.inputs["Vector"])
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.12; L(vo.outputs["Distance"], lt.inputs[0])
        base = p.inputs["Base Color"].links[0].from_socket
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; L(lt.outputs[0], mx.inputs["Factor"]); L(base, mx.inputs["A"])
        mx.inputs["B"].default_value = (0.55, 0.18, 0.02, 1)
        L(mx.outputs["Result"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.45
    trees = []
    for v in range(3):
        t = W.crown_proxy(f"Citrus{v}", src, 300 + v)
        t.data.materials[0] = leaf
        t.hide_render = True
        trees.append(t)
    pts = []
    order = rng.permutation(len(hamlets))
    for hi in order[:n_sites]:
        hs, hl, hx, hy = hamlets[hi]
        for attempt in range(30):
            a = rng.uniform(0, 2 * math.pi); r = rng.uniform(110, 240)
            cx, cy = hx + r * math.cos(a), hy + r * math.sin(a)
            if _free(cx, cy, 32.0):
                break
        else:
            continue
        yaw = rng.uniform(0, math.pi)
        W_, D_ = rng.uniform(40, 70), rng.uniform(30, 50)
        u = np.arange(-W_ / 2, W_ / 2, 3.6); v = np.arange(-D_ / 2, D_ / 2, 4.6)
        U, Vv = np.meshgrid(u, v); U = U.ravel() + rng.normal(0, 0.15, U.size); Vv = Vv.ravel()
        xs = cx + U * math.cos(yaw) - Vv * math.sin(yaw); ys = cy + U * math.sin(yaw) + Vv * math.cos(yaw)
        keep = rng.random(len(xs)) < 0.93
        xs, ys = xs[keep], ys[keep]
        pts.append(np.column_stack([xs, ys, W.natural_ground(xs, ys)]))
        KEEPOUT.append((cx, cy, math.hypot(W_, D_) / 2 + 2))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    W.scatter_tree("PropOrchards", coll, P, trees, (1.0, 1.35), 41, wind=0.0)
    return len(P)


def tombs(coll, rng, n=26):
    """Family tombs on the slopes above the cutting: white tiled/concrete front wall with a stele, a curved
    back wall round the grave mound, a small paved forecourt, facing downhill."""
    white = M.principled("TombWhite", (0.62, 0.62, 0.60), 0.0, 0.5)
    stele = M.principled("TombStele", (0.06, 0.06, 0.065), 0.0, 0.25, coat=0.5)
    soil = M.principled("TombMound", (0.09, 0.10, 0.05), 0.0, 0.95)
    mb = MB()
    mb.box(-1.3, 1.3, -0.25, 0.0, -0.4, 1.25, white)                 # front wall
    mb.box(-0.42, 0.42, -0.30, -0.25, 0.25, 1.05, stele)
    mb.box(-1.5, 1.5, -2.2, -0.25, -0.4, 0.08, white)                 # forecourt
    for k in range(9):                                                 # horseshoe back wall
        a0 = math.pi * k / 9; a1 = math.pi * (k + 1) / 9
        p0 = (1.3 * math.cos(a0), 1.6 * math.sin(a0)); p1 = (1.3 * math.cos(a1), 1.6 * math.sin(a1))
        mb.box(min(p0[0], p1[0]) - 0.1, max(p0[0], p1[0]) + 0.1, min(p0[1], p1[1]), max(p0[1], p1[1]) + 0.2, -0.4, 0.9, white)
    mb.cyl((0, 0.7, -0.3), (0, 0.7, 0.55), 1.05, soil, segs=12)       # grave mound
    mats = [white, stele, soil]; idx = {m.name: i for i, m in enumerate(mats)}
    mb.m = [idx[m.name] for m in mb.m]
    me = mb.make("TombMesh", mats)
    placed = 0
    tries = 0
    while placed < n and tries < 4000:
        tries += 1
        s = rng.uniform(3700, A.TUNNEL_S - 30); lat = rng.choice([-1, 1]) * rng.uniform(30, 140)
        x, y = _place_xy(np.array([s]), np.array([lat]))
        x, y = float(x[0]), float(y[0])
        g = float(W.ground(np.array([x]), np.array([y]))[0]); nat = float(W.natural_ground(np.array([x]), np.array([y]))[0])
        if nat - g > 0.2 or g < A.RAIL_Z + 4.0:                       # not in the cutting / not on the valley floor
            continue
        e = 2.0
        gx = float(W.ground(np.array([x + e]), np.array([y]))[0] - W.ground(np.array([x - e]), np.array([y]))[0]) / (2 * e)
        gy = float(W.ground(np.array([x]), np.array([y + e]))[0] - W.ground(np.array([x]), np.array([y - e]))[0]) / (2 * e)
        slope = math.hypot(gx, gy)
        if slope < 0.08 or slope > 0.7:
            continue
        downhill = math.atan2(-gy, -gx)
        for k in range(int(rng.integers(1, 4))):                     # family groups of 1-3
            ox, oy = rng.normal(0, 3.5, 2)
            ob = bpy.data.objects.new(f"Tomb{placed:02d}", me); coll.objects.link(ob)
            gz = float(W.ground(np.array([x + ox]), np.array([y + oy]))[0])
            ob.location = (x + ox, y + oy, gz); ob.rotation_euler = (0, 0, downhill + math.pi / 2 + rng.normal(0, 0.15))
            KEEPOUT.append((x + ox, y + oy, 3.5))
            placed += 1
    return placed


def build_props(coll, src, hamlets, seed=55):
    rng = np.random.default_rng(seed)
    _, npoles = power_lines(coll)
    ponds = fish_ponds(coll, hamlets, rng)
    gh = greenhouses(coll, hamlets, rng)
    nst = straw_stacks(coll, src, hamlets, rng)
    no = orchards(coll, src, hamlets, rng)
    nt = tombs(coll, rng)
    print(f"[build]   props: {npoles} poles, {len(ponds)} fish ponds, {len(gh)} greenhouse sites, {nst} straw stacks, "
          f"{no} orchard trees, {nt} tombs", flush=True)
