"""High-speed railway infrastructure along the alignment (bpy):
32 m box-girder viaduct on round-ended piers, CRTS III slab track with CHN60 rails and WJ-8 fasteners,
cable troughs, parapets, noise barriers, OCS masts every 48 m with cantilevers, zig-zag contact wire
(+-300 mm), messenger with sag and droppers, feeder / protection wires, tunnel portal, signs, GSM-R mast.
"""
import math

import bpy
import numpy as np

from . import alignment as A
from . import materials as M
from . import specs as S
from .bl import new_mesh_object, grid_faces, instancer, matrix_to_euler

GIRDER_TOP_BELOW_RAIL = 0.72


# ------------------------------------------------------------------ generic sweep
def _frames_flat(s):
    """Horizontal (uncanted) frames on the centre line at rail level."""
    th = A.heading(s); x, y = A.xy(s)
    T = np.stack([np.cos(th), np.sin(th), 0 * th], -1)
    N = np.stack([-np.sin(th), np.cos(th), 0 * th], -1)
    U = np.stack([0 * th, 0 * th, 1 + 0 * th], -1)
    O = np.stack([x, y, A.rail_z(s)], -1)
    return O, T, N, U


def sweep(name, prof, s0, s1, ds=2.0, canted_offset=None, closed=True, coll=None, mat=None, smooth=False, uv_scale=1.0, cap=False):
    """Sweep a 2D profile [(lat, up), ...] along the alignment from s0 to s1.

    canted_offset: None -> structure (no cant, centre line); float -> track frame with cant at that offset.
    UVs: u = arc length along profile, v = chainage (metres * uv_scale).
    """
    n_s = max(2, int(math.ceil((s1 - s0) / ds)) + 1)
    s = np.linspace(s0, s1, n_s)
    if canted_offset is None:
        O, T, N, U = _frames_flat(s)
    else:
        O, T, N, U = A.frame(s, canted_offset)
    prof = np.asarray(prof, float)
    P = O[:, None, :] + N[:, None, :] * prof[None, :, 0:1] + U[:, None, :] * prof[None, :, 1:2]
    verts = P.reshape(-1, 3)
    m = len(prof)
    faces = grid_faces(n_s, m, closed=closed)
    seg = np.linalg.norm(np.diff(np.vstack([prof, prof[:1]]) if closed else prof, axis=0), axis=1)
    ua = np.concatenate([[0], np.cumsum(seg)])
    uvs = []
    for i in range(n_s - 1):
        for j in range(m if closed else m - 1):
            j2 = j + 1
            uvs += [(ua[j], s[i] * uv_scale), (ua[j2], s[i] * uv_scale), (ua[j2], s[i + 1] * uv_scale), (ua[j], s[i + 1] * uv_scale)]
    if cap and closed:
        for r in (0, n_s - 1):
            c = len(verts)
            verts = np.vstack([verts, P[r].mean(0)])
            for j in range(m):
                j2 = (j + 1) % m
                faces.append((r * m + j, r * m + j2, c) if r else (r * m + j2, r * m + j, c))
                uvs += [(0, 0), (0, 0), (0, 0)]
    ob = new_mesh_object(name, verts, faces, coll, uvs=np.array(uvs), smooth=smooth)
    if mat:
        ob.data.materials.append(mat)
    return ob


def place(ob, s, offset=0.0, canted=False, up=0.0, yaw=0.0):
    if canted:
        O, T, N, U = A.frame(s, offset)
    else:
        O, T, N, U = _frames_flat(np.atleast_1d(s)); O, T, N, U = O[0], T[0], N[0], U[0]
        O = O + N * offset
    from mathutils import Matrix
    ob.matrix_world = Matrix(((T[0], N[0], U[0], O[0] + U[0] * up), (T[1], N[1], U[1], O[1] + U[1] * up),
                              (T[2], N[2], U[2], O[2] + U[2] * up), (0, 0, 0, 1)))
    if yaw:
        ob.rotation_euler.z += yaw


# ------------------------------------------------------------------ materials
def concrete(name="Concrete", tex=None, tint=(1, 1, 1), scale=0.25, rough_add=0.0, cell=None, streaks=True, hab=False):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (scale, scale, scale)
    L(tc.outputs["UV"], mp.inputs["Vector"])
    if tex:
        d = nt.nodes.new("ShaderNodeTexImage"); d.image = bpy.data.images.load(tex["diff"], check_existing=True)
        r = nt.nodes.new("ShaderNodeTexImage"); r.image = bpy.data.images.load(tex["rough"], check_existing=True); r.image.colorspace_settings.name = "Non-Color"
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(tex["nor"], check_existing=True); n.image.colorspace_settings.name = "Non-Color"
        for t in (d, r, n):
            L(mp.outputs[0], t.inputs[0])
        # large-scale staining (water streaks, efflorescence) to break tiling
        st = nt.nodes.new("ShaderNodeTexNoise"); st.inputs["Scale"].default_value = 0.6; st.inputs["Detail"].default_value = 6
        L(tc.outputs["Object"], st.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs[1].default_value = 0.35; mr.inputs[2].default_value = 0.75
        mr.inputs[3].default_value = 0.78; mr.inputs[4].default_value = 1.12
        L(st.outputs["Fac"], mr.inputs[0])
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs["Factor"].default_value = 1.0
        tintn = nt.nodes.new("ShaderNodeMix"); tintn.data_type = "RGBA"; tintn.blend_type = "MULTIPLY"; tintn.inputs["Factor"].default_value = 1.0
        tintn.inputs["B"].default_value = (*tint, 1)
        hs = nt.nodes.new("ShaderNodeHueSaturation"); hs.inputs["Saturation"].default_value = 0.35; hs.inputs["Value"].default_value = 1.15
        L(d.outputs["Color"], hs.inputs["Color"]); L(hs.outputs["Color"], tintn.inputs["A"]); L(tintn.outputs["Result"], mul.inputs["A"])
        cmb = nt.nodes.new("ShaderNodeCombineColor"); L(mr.outputs[0], cmb.inputs[0]); L(mr.outputs[0], cmb.inputs[1]); L(mr.outputs[0], cmb.inputs[2])
        L(cmb.outputs[0], mul.inputs["B"])
        col = mul.outputs["Result"]
        rough = r.outputs["Color"]
        col, rough = _weathering(nt, col, rough, cell, streaks, hab)
        L(col, p.inputs["Base Color"])
        ra = nt.nodes.new("ShaderNodeMath"); ra.operation = "ADD"; ra.inputs[1].default_value = rough_add
        L(rough, ra.inputs[0]); L(ra.outputs[0], p.inputs["Roughness"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.8
        L(n.outputs["Color"], nm.inputs["Color"]); L(nm.outputs[0], p.inputs["Normal"])
    else:
        p.inputs["Base Color"].default_value = (0.5 * tint[0], 0.5 * tint[1], 0.48 * tint[2], 1)
        p.inputs["Roughness"].default_value = 0.85
    L(p.outputs[0], out.inputs[0])
    return m



def _weathering(nt, col, rough, cell=None, streaks=True, hab=False):
    """Real-concrete weathering on top of the photo texture (all cheap procedural nodes, no extra rays):
    - element-to-element tone (each 32 m girder span / parapet segment by chainage, each pier by world cell),
    - rain-runoff streaks on vertical faces (dark dirt streaks, a few pale efflorescence streaks), patchy,
    - splash / soil / moss zone at the foot of the piers (vertex attribute 'hab' = height above ground)."""
    L = nt.links.new
    def node(t, **kw):
        n = nt.nodes.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n
    def math(op, a, b=None, clamp=False):
        n = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                L(x, n.inputs[i])
        return n.outputs[0]
    def scale_col(c, f):
        n = node("ShaderNodeVectorMath", operation="SCALE"); L(c, n.inputs[0]); L(f, n.inputs["Scale"]); return n.outputs[0]
    tc = node("ShaderNodeTexCoord"); geo = node("ShaderNodeNewGeometry")
    if cell == "uv32":
        sep = node("ShaderNodeSeparateXYZ"); L(tc.outputs["UV"], sep.inputs[0])
        k = math("FLOOR", math("DIVIDE", sep.outputs["Y"], 32.0))
        wn = node("ShaderNodeTexWhiteNoise", noise_dimensions="1D"); L(k, wn.inputs["W"])
        col = scale_col(col, math("ADD", math("MULTIPLY", wn.outputs["Value"], 0.16), 0.92))
    elif cell == "world":
        vo = node("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = 1.0 / 30.0
        L(geo.outputs["Position"], vo.inputs["Vector"])
        sepc = node("ShaderNodeSeparateColor"); L(vo.outputs["Color"], sepc.inputs[0])
        col = scale_col(col, math("ADD", math("MULTIPLY", sepc.outputs[0], 0.16), 0.92))
    if streaks:
        sepn = node("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], sepn.inputs[0])
        vert = math("SUBTRACT", 1.0, math("ABSOLUTE", sepn.outputs["Z"]))
        vmask = node("ShaderNodeMapRange"); L(vert, vmask.inputs[0]); vmask.inputs[1].default_value = 0.75; vmask.inputs[2].default_value = 0.97
        mp = node("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (2.6, 2.6, 0.07)
        L(geo.outputs["Position"], mp.inputs["Vector"])
        nz = node("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.0; nz.inputs["Detail"].default_value = 5.0
        L(mp.outputs[0], nz.inputs["Vector"])
        dark = node("ShaderNodeMapRange"); L(nz.outputs["Fac"], dark.inputs[0]); dark.inputs[1].default_value = 0.52; dark.inputs[2].default_value = 0.80
        pale = node("ShaderNodeMapRange"); L(nz.outputs["Fac"], pale.inputs[0]); pale.inputs[1].default_value = 0.30; pale.inputs[2].default_value = 0.16
        patch = node("ShaderNodeTexNoise"); patch.inputs["Scale"].default_value = 0.035; patch.inputs["Detail"].default_value = 2.0
        L(geo.outputs["Position"], patch.inputs["Vector"])
        pm = node("ShaderNodeMapRange"); L(patch.outputs["Fac"], pm.inputs[0]); pm.inputs[1].default_value = 0.35; pm.inputs[2].default_value = 0.65
        amt = math("MULTIPLY", vmask.outputs[0], math("ADD", math("MULTIPLY", pm.outputs[0], 0.7), 0.3))
        d = math("MULTIPLY", math("MULTIPLY", dark.outputs[0], amt), 0.30)
        e = math("MULTIPLY", math("MULTIPLY", pale.outputs[0], amt), 0.10)
        col = scale_col(col, math("ADD", math("SUBTRACT", 1.0, d), e))
        rough = math("ADD", rough, math("MULTIPLY", d, -0.3))       # wet-dark streaks are slightly smoother
    if hab:
        at = node("ShaderNodeAttribute", attribute_name="hab")
        foot = node("ShaderNodeMapRange"); L(at.outputs["Fac"], foot.inputs[0])
        foot.inputs[1].default_value = 2.2; foot.inputs[2].default_value = 0.0
        soil = node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        L(foot.outputs[0], soil.inputs["Factor"]); L(col, soil.inputs["A"]); soil.inputs["B"].default_value = (0.55, 0.52, 0.40, 1)
        col = soil.outputs["Result"]
    return col, rough

# ------------------------------------------------------------------ profiles
def girder_profile():
    """Single-cell box girder 12.6 m wide, 3.05 m deep (z relative to girder top)."""
    h = S.GIRDER_DEPTH; w = S.GIRDER_TOP_W / 2
    right = [(-w, 0.0), (-w, -0.22), (-w + 0.15, -0.26), (-3.55, -0.55), (-3.25, -0.80), (-2.80, -h), ]
    left = [(-y, z) for (y, z) in reversed(right)]
    return right + left


def parapet_profile(side):
    """Outer edge: cable trough + parapet wall (lat sign = side)."""
    w = S.GIRDER_TOP_W / 2
    pts = [(4.25, 0.0), (4.25, 0.55), (4.42, 0.55), (4.42, 0.12), (5.85, 0.12), (5.85, 0.55), (6.0, 0.55), (6.0, 1.15),
           (6.3, 1.15), (6.3, 0.0)]
    return [(side * a, b) for a, b in (pts if side > 0 else reversed(pts))]


def baseplate_profile():
    return [(-1.45, 0.0), (1.45, 0.0), (1.45, 0.25), (1.35, 0.30), (-1.35, 0.30), (-1.45, 0.25)][::-1]


def rail_profile():
    """CHN60 rail (60 kg/m), 176 mm high, 73 mm head, 150 mm foot; local lat, up (up=0 at rail top)."""
    p = [(-0.075, -0.176), (0.075, -0.176), (0.075, -0.165), (0.020, -0.150), (0.0085, -0.120), (0.0085, -0.060),
         (0.034, -0.040), (0.0365, -0.010), (0.030, 0.0), (-0.030, 0.0), (-0.0365, -0.010), (-0.034, -0.040),
         (-0.0085, -0.060), (-0.0085, -0.120), (-0.020, -0.150), (-0.075, -0.165)]
    return p[::-1]


# ------------------------------------------------------------------ builders
def build_viaduct(coll, tex):
    con = concrete("ViaductConcrete", tex.get("concrete"), (0.95, 0.94, 0.90), 0.22, cell="uv32")
    s0, s1 = A.VIADUCT
    spans = np.arange(s0, s1 - 1, S.GIRDER_SPAN)
    objs = []
    gp = [(y, z - GIRDER_TOP_BELOW_RAIL) for y, z in girder_profile()]
    for k, sa in enumerate(spans):
        objs.append(sweep(f"Girder{k:03d}", gp, sa + 0.05, sa + S.GIRDER_SPAN - 0.05, 4.0, coll=coll, mat=con, cap=True, uv_scale=1.0))
    for side in (1, -1):
        pp = [(y, z - GIRDER_TOP_BELOW_RAIL) for y, z in parapet_profile(side)]
        objs.append(sweep(f"Parapet{side}", pp, s0, s1, 4.0, coll=coll, mat=con, closed=True))
    return objs


def build_piers(coll, tex, ground_fn):
    con = concrete("PierConcrete", tex.get("concrete"), (0.92, 0.91, 0.88), 0.2, cell="world", hab=True)
    s0, s1 = A.VIADUCT
    joints = np.arange(s0 + S.GIRDER_SPAN, s1 - 1, S.GIRDER_SPAN)
    verts, faces, habs = [], [], []
    top = A.RAIL_Z - GIRDER_TOP_BELOW_RAIL - S.GIRDER_DEPTH - 0.25      # underside of bearings
    nseg = 12
    for s in joints:
        O, T, N, U = _frames_flat(np.array([s])); O, T, N = O[0], T[0], N[0]
        g = ground_fn(O[0], O[1])
        H = top - g
        if H < 1.0:
            continue
        cap_h = 1.6
        rings = []
        for z, hw, hl in ((g - 2.0, 3.6, 1.6), (top - cap_h, 3.0 + 0.0125 * 0, 1.3), (top - cap_h, 3.0, 1.3),
                          (top - cap_h + 0.4, 3.8, 1.7), (top, 3.8, 1.7)):
            ring = []
            for half in (1, -1):
                for k in range(nseg + 1):
                    a = -math.pi / 2 + math.pi * k / nseg
                    lat = half * (hw - hl + hl * math.cos(a))
                    lon = hl * math.sin(a) * half
                    ring.append(O + N * lat + T * lon + np.array([0, 0, z - O[2]]))
            rings.append(ring)
        # taper: body width grows 1:40 downwards
        body_w = 3.0 + H / 40 / 2
        rings[0] = [r for r in rings[0]]
        base = len(verts)
        rl = len(rings[0])
        for ri, ring in enumerate(rings):
            if ri == 0:
                ring = [O + (p - O) * np.array([1, 1, 1]) for p in ring]
            verts.extend(ring)
            habs.extend([float(p[2] - g) for p in ring])
        for ri in range(len(rings) - 1):
            for j in range(rl):
                j2 = (j + 1) % rl
                faces.append((base + ri * rl + j, base + ri * rl + j2, base + (ri + 1) * rl + j2, base + (ri + 1) * rl + j))
        c = len(verts); verts.append(np.mean(rings[-1], axis=0)); habs.append(float(top - g))
        for j in range(rl):
            faces.append((base + (len(rings) - 1) * rl + j, base + (len(rings) - 1) * rl + (j + 1) % rl, c))
    ob = new_mesh_object("Piers", np.array(verts), faces, coll, smooth=True)
    at = ob.data.attributes.new("hab", "FLOAT", "POINT")                 # height above the ground at that pier
    at.data.foreach_set("value", np.asarray(habs, np.float32))
    _box_uv(ob)
    ob.data.materials.append(con)
    return ob


def _box_uv(ob, scale=1.0):
    me = ob.data
    uv = me.uv_layers.new(name="UVMap")
    co = np.array([v.co for v in me.vertices])
    data = []
    for p in me.polygons:
        n = np.abs(np.array(p.normal))
        ax = int(np.argmax(n))
        for li in p.loop_indices:
            v = co[me.loops[li].vertex_index]
            a, b = [(1, 2), (0, 2), (0, 1)][ax]
            data.append((v[a] * scale, v[b] * scale))
    uv.data.foreach_set("uv", np.array(data, np.float32).ravel())


def build_track(coll, tex, s0, s1):
    """Base plates, SCC layer, CRTS III slabs, rails, fasteners for both tracks."""
    con = concrete("SlabConcrete", tex.get("slab"), (0.97, 0.96, 0.93), 0.3, streaks=False)
    base_m = concrete("BaseConcrete", tex.get("concrete"), (0.80, 0.79, 0.76), 0.3, 0.05, streaks=False)
    rail_m = M.rail_material()
    fast_m = M.principled("Fastener", (0.05, 0.05, 0.05), 0.4, 0.55)
    clip_m = M.principled("FastenerClip", (0.30, 0.31, 0.32), 1.0, 0.45)
    objs = []
    slab_src = _slab_template(coll, con)
    fast_src = _fastener_template(coll, fast_m, clip_m)
    for tr in (+1, -1):
        off = tr * S.TRACK_CENTRES / 2
        bp = [(y, z - 0.72) for y, z in baseplate_profile()]
        objs.append(sweep(f"BasePlate{tr}", bp, s0, s1, 3.0, canted_offset=off, coll=coll, mat=base_m))
        # slabs
        pitch = S.SLAB_L + S.SLAB_GAP
        ss = np.arange(s0 + 1.0, s1 - pitch, pitch)
        O, T, N, U = A.frame(ss + S.SLAB_L / 2, off)
        rot = matrix_to_euler(T, N, U)
        objs.append(instancer(f"Slabs{tr}", slab_src, O + U * (-0.38), rot, coll=coll))
        # fasteners: 9 per slab per rail
        fs = (ss[:, None] + 0.36 + np.arange(S.FASTENERS_PER_SLAB)[None, :] * 0.61).ravel()
        for rside in (1, -1):
            O, T, N, U = A.frame(fs, off)
            pos = O + N * (rside * 0.7175) + U * (-0.176)
            rot = matrix_to_euler(T, N * rside, U)
            objs.append(instancer(f"Fast{tr}{rside}", fast_src, pos, rot, coll=coll))
        # rails (1:40 inclination towards the track centre)
        for rside in (1, -1):
            inc = -rside * S.RAIL_INCLINATION
            prof = [(rside * 0.7175 + y + inc * z, z) for y, z in rail_profile()]
            objs.append(sweep(f"Rail{tr}{rside}", prof, s0, s1, 2.0, canted_offset=off, coll=coll, mat=rail_m, smooth=True))
    for o in (slab_src, fast_src):
        o.hide_render = True; o.hide_viewport = True
    return objs


def _slab_template(coll, mat):
    from .trainparts import box, join, setmat
    parts = [setmat(box("slab", (S.SLAB_L, S.SLAB_W, S.SLAB_T), (0, 0, 0.1), coll, 0.01), mat)]
    for k in range(S.FASTENERS_PER_SLAB):         # rail seat pads (raised bearing platforms)
        x = -S.SLAB_L / 2 + 0.36 + k * 0.61
        for y in (-0.7175, 0.7175):
            parts.append(setmat(box("seat", (0.32, 0.38, 0.03), (x, y, 0.215), coll), mat))
    ob = join(parts, "SlabTemplate"); coll.objects.link(ob)
    return ob


def _fastener_template(coll, mat, clip):
    """WJ-8 style: base plate + two W-shaped tension clamps + bolts (origin under rail foot centre)."""
    from .trainparts import box, join, setmat, tube_between
    parts = [setmat(box("pad", (0.18, 0.40, 0.02), (0, 0, -0.01), coll), mat)]
    for y in (-0.13, 0.13):
        parts.append(setmat(box("iso", (0.12, 0.06, 0.04), (0, y, 0.01), coll), mat))
        pts = [(-0.06, y * 1.25, 0.01), (-0.05, y * 0.95, 0.045), (0.0, y * 0.78, 0.03), (0.05, y * 0.95, 0.045), (0.06, y * 1.25, 0.01)]
        for a, b in zip(pts[:-1], pts[1:]):
            parts.append(setmat(tube_between("clip", a, b, 0.0065, coll, 6), clip))
        parts.append(setmat(tube_between("bolt", (0, y * 1.35, 0.0), (0, y * 1.35, 0.06), 0.012, coll, 6), clip))
    ob = join(parts, "FastenerTemplate"); coll.objects.link(ob)
    return ob


def build_noise_barriers(coll, s0, s1):
    post_m = M.principled("BarrierPost", (0.28, 0.30, 0.32), 0.8, 0.5)
    panel_m = M.principled("BarrierPanel", (0.62, 0.64, 0.63), 0.3, 0.55)
    pc_m = M.principled("BarrierPC", (0.55, 0.62, 0.62), 0.0, 0.08, alpha=0.35)
    objs = []
    gtop = -GIRDER_TOP_BELOW_RAIL
    for side in (1, -1):
        y0 = side * 6.12
        prof = [(y0 - 0.05, 1.15 + gtop), (y0 + 0.05, 1.15 + gtop), (y0 + 0.05, S.NOISE_BARRIER_H - 0.7), (y0 - 0.05, S.NOISE_BARRIER_H - 0.7)]
        objs.append(sweep(f"BarrierPanel{side}", prof, s0, s1, 4.0, coll=coll, mat=panel_m))
        prof2 = [(y0 - 0.02, S.NOISE_BARRIER_H - 0.7), (y0 + 0.02, S.NOISE_BARRIER_H - 0.7), (y0 + 0.02, S.NOISE_BARRIER_H), (y0 - 0.02, S.NOISE_BARRIER_H)]
        objs.append(sweep(f"BarrierPC{side}", prof2, s0, s1, 4.0, coll=coll, mat=pc_m))
        from .trainparts import box, setmat
        src = setmat(box(f"PostSrc{side}", (0.20, 0.20, S.NOISE_BARRIER_H - gtop - 1.0), (0, 0, (S.NOISE_BARRIER_H + gtop + 1.15) / 2), coll), post_m)
        src.hide_render = True; src.hide_viewport = True
        ss = np.arange(s0, s1, 2.0)
        O, T, N, U = _frames_flat(ss)
        objs.append(instancer(f"BarrierPosts{side}", src, O + N * (y0 + side * 0.1), matrix_to_euler(T, N, U), coll=coll))
    return objs


# ------------------------------------------------------------------ overhead contact system
def mast_chainages(s0, s1):
    return np.arange(s0, s1, S.MAST_SPACING_VIADUCT)


def contact_points(track_off, s_masts):
    """Contact wire support points (world) with alternating +-300 mm stagger, in the canted track frame."""
    O, T, N, U = A.frame(s_masts, track_off)
    stg = S.STAGGER * np.where(np.arange(len(s_masts)) % 2 == 0, 1.0, -1.0)
    cw = O + N * stg[:, None] + U * S.CONTACT_WIRE_HEIGHT
    return cw, stg, (O, T, N, U)


def wire_polyline(track_off, s_masts, ds=0.5, kind="contact"):
    """Dense polyline (points, chainage) for the contact or messenger wire of one track."""
    cw, stg, _ = contact_points(track_off, s_masts)
    pts, ss = [], []
    for i in range(len(s_masts) - 1):
        sa, sb = s_masts[i], s_masts[i + 1]
        n = int(round((sb - sa) / ds))
        for k in range(n):
            f = k / n
            s = sa + f * (sb - sa)
            O, T, N, U = A.frame(np.array([s]), track_off)
            lat = stg[i] * (1 - f) + stg[i + 1] * f
            h = S.CONTACT_WIRE_HEIGHT
            if kind == "messenger":
                sag = 4 * 0.62 * f * (1 - f)              # 0.62 m sag of the messenger at mid span
                h = S.CONTACT_WIRE_HEIGHT + S.SYSTEM_HEIGHT - sag
            pts.append(O[0] + N[0] * lat + U[0] * h); ss.append(s)
    return np.array(pts), np.array(ss)


def tube_mesh(name, pts, r, coll, sides=6, mat=None, attr_s=None):
    pts = np.asarray(pts, float)
    t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    up = np.array([0, 0, 1.0])
    b1 = np.cross(t, up); b1 /= np.linalg.norm(b1, axis=1, keepdims=True) + 1e-12
    b2 = np.cross(b1, t)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    V = pts[:, None, :] + r * (np.cos(ang)[None, :, None] * b1[:, None, :] + np.sin(ang)[None, :, None] * b2[:, None, :])
    faces = grid_faces(len(pts), sides, closed=True)
    ob = new_mesh_object(name, V.reshape(-1, 3), faces, coll, smooth=True)
    if attr_s is not None:
        a = ob.data.attributes.new("s", "FLOAT", "POINT")
        a.data.foreach_set("value", np.repeat(np.asarray(attr_s, np.float32), sides))
    if mat:
        ob.data.materials.append(mat)
    return ob


def uplift_tree():
    """GN: z += uplift(s - s_panto) in the local up direction.  Inputs: Panto S, Amplitude."""
    ng = bpy.data.node_groups.get("WireUplift")
    if ng:
        return ng
    ng = bpy.data.node_groups.new("WireUplift", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_in = ng.interface.new_socket("Panto S", in_out="INPUT", socket_type="NodeSocketFloat"); s_in.default_value = -1e6
    a_in = ng.interface.new_socket("Amplitude", in_out="INPUT", socket_type="NodeSocketFloat"); a_in.default_value = 0.12
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; L = ng.links.new
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    na = N.new("GeometryNodeInputNamedAttribute"); na.data_type = "FLOAT"; na.inputs["Name"].default_value = "s"
    d = N.new("ShaderNodeMath"); d.operation = "SUBTRACT"
    L(na.outputs["Attribute"], d.inputs[0]); L(gi.outputs["Panto S"], d.inputs[1])
    # asymmetric bump: ahead (d>0) decays over 9 m, behind over 22 m with a damped ripple
    ab = N.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; L(d.outputs[0], ab.inputs[0])
    gt = N.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.0; L(d.outputs[0], gt.inputs[0])
    lam = N.new("ShaderNodeMix"); lam.data_type = "FLOAT"; lam.inputs["A"].default_value = 22.0; lam.inputs["B"].default_value = 9.0
    L(gt.outputs[0], lam.inputs["Factor"])
    q = N.new("ShaderNodeMath"); q.operation = "DIVIDE"; L(ab.outputs[0], q.inputs[0]); L(lam.outputs["Result"], q.inputs[1])
    sq = N.new("ShaderNodeMath"); sq.operation = "MULTIPLY"; L(q.outputs[0], sq.inputs[0]); L(q.outputs[0], sq.inputs[1])
    ng_ = N.new("ShaderNodeMath"); ng_.operation = "MULTIPLY"; ng_.inputs[1].default_value = -1.0; L(sq.outputs[0], ng_.inputs[0])
    ex = N.new("ShaderNodeMath"); ex.operation = "EXPONENT"; L(ng_.outputs[0], ex.inputs[0])
    rip = N.new("ShaderNodeMath"); rip.operation = "COSINE"
    rs = N.new("ShaderNodeMath"); rs.operation = "MULTIPLY"; rs.inputs[1].default_value = 2 * math.pi / 26.0
    L(d.outputs[0], rs.inputs[0]); L(rs.outputs[0], rip.inputs[0])
    rmix = N.new("ShaderNodeMath"); rmix.operation = "MULTIPLY_ADD"; rmix.inputs[1].default_value = 0.25; rmix.inputs[2].default_value = 0.75
    L(rip.outputs[0], rmix.inputs[0])
    m1 = N.new("ShaderNodeMath"); m1.operation = "MULTIPLY"; L(ex.outputs[0], m1.inputs[0]); L(rmix.outputs[0], m1.inputs[1])
    m2 = N.new("ShaderNodeMath"); m2.operation = "MULTIPLY"; L(m1.outputs[0], m2.inputs[0]); L(gi.outputs["Amplitude"], m2.inputs[1])
    cv = N.new("ShaderNodeCombineXYZ"); L(m2.outputs[0], cv.inputs["Z"])
    sp = N.new("GeometryNodeSetPosition"); L(gi.outputs["Geometry"], sp.inputs["Geometry"]); L(cv.outputs[0], sp.inputs["Offset"])
    L(sp.outputs[0], go.inputs["Geometry"])
    return ng


def build_ocs(coll, s0, s1):
    """Masts, cantilevers, wires for both tracks.  Returns dict with contact-wire objects (for uplift keys)."""
    from .trainparts import box, join, setmat, tube_between
    steel = M.principled("MastSteel", (0.42, 0.43, 0.44), 0.9, 0.45)
    galv = M.principled("Galvanised", (0.55, 0.56, 0.57), 1.0, 0.38)
    alu = M.principled("CantileverAlu", (0.70, 0.71, 0.72), 1.0, 0.3)
    insul = M.principled("OCSInsulator", (0.32, 0.20, 0.12), 0.0, 0.3, coat=0.5)
    cu = M.principled("CopperWire", (0.58, 0.36, 0.22), 1.0, 0.33)
    blk = M.principled("FeederWire", (0.18, 0.18, 0.19), 1.0, 0.5)
    s_m = mast_chainages(s0, s1)
    res = dict(contact=[], messenger=[], masts=[])
    for tr in (+1, -1):
        off = tr * S.TRACK_CENTRES / 2
        cw, stg, (O, T, N, U) = contact_points(off, s_m)
        mast_lat = tr * 5.55          # mast on the outer deck edge of its track
        Of, Tf, Nf, Uf = _frames_flat(s_m)
        deck = A.RAIL_Z - GIRDER_TOP_BELOW_RAIL
        parts = []
        for i in range(len(s_m)):
            base = Of[i] + Nf[i] * mast_lat
            base[2] = deck
            topz = A.RAIL_Z + 8.6
            # H-beam mast (two flanges + web)
            for dl in (-0.13, 0.13):
                p0 = base + Tf[i] * dl; p1 = p0.copy(); p1[2] = topz
                parts.append(setmat(tube_between("fl", p0, p1, 0.022, coll, 4), steel))
            p0 = base.copy(); p1 = base.copy(); p1[2] = topz
            parts.append(setmat(tube_between("web", p0, p1, 0.09, coll, 4), steel))
            # cantilever: flat tube at messenger height, stay tube, registration tube + steady arm
            wire = cw[i]
            msg = wire + U[i] * S.SYSTEM_HEIGHT
            mast_top_att = base.copy(); mast_top_att[2] = msg[2] + 0.05
            mast_low_att = base.copy(); mast_low_att[2] = wire[2] - 0.45
            toward = -Nf[i] * tr
            parts.append(setmat(tube_between("ins1", mast_top_att, mast_top_att + toward * 0.55, 0.045, coll, 8), insul))
            parts.append(setmat(tube_between("ins2", mast_low_att, mast_low_att + toward * 0.55 + np.array([0, 0, 0.15]), 0.045, coll, 8), insul))
            parts.append(setmat(tube_between("flat", mast_top_att + toward * 0.55, msg + toward * 0.25, 0.028, coll, 8), alu))
            stay_end = msg + toward * 0.2 + np.array([0, 0, -0.05])
            parts.append(setmat(tube_between("stay", mast_low_att + toward * 0.55 + np.array([0, 0, 0.15]), stay_end, 0.025, coll, 8), alu))
            reg_a = mast_low_att + toward * 0.55 + np.array([0, 0, 0.15]) + (stay_end - (mast_low_att + toward * 0.55)) * 0.35
            reg_b = wire + U[i] * 0.32 + (-toward) * (0.9 * (1 if stg[i] * tr > 0 else -1)) * 0.0 + toward * 0.6
            parts.append(setmat(tube_between("reg", reg_a, reg_b, 0.02, coll, 8), alu))
            # steady arm (pulls the wire to its stagger), angled down onto the contact wire
            sa_a = reg_b; sa_b = wire + U[i] * 0.03
            parts.append(setmat(tube_between("steady", sa_a, sa_b, 0.013, coll, 6), alu))
            # feeder (AF) on top with insulator, protection wire bracket on field side
            parts.append(setmat(tube_between("fbr", base + np.array([0, 0, topz - deck - 0.1]), base + np.array([0, 0, topz - deck - 0.1]) - toward * 0.8, 0.025, coll, 6), galv))
            parts.append(setmat(tube_between("fins", base + np.array([0, 0, topz - deck - 0.1]) - toward * 0.8, base + np.array([0, 0, topz - deck + 0.35]) - toward * 0.8, 0.05, coll, 8), insul))
            parts.append(setmat(box("plate", (0.02, 0.18, 0.26), tuple(base + np.array([0, 0, 2.6]) + toward * 0.12), coll), M.principled("MastPlate", (0.9, 0.9, 0.88), 0, 0.5)))
        mast_ob = join(parts, f"OCS_Masts{tr}"); coll.objects.link(mast_ob)
        res["masts"].append(mast_ob)
        # wires
        cpts, cs = wire_polyline(off, s_m, 0.5, "contact")
        cob = tube_mesh(f"ContactWire{tr}", cpts, 0.0072, coll, 6, cu, attr_s=cs)
        md = cob.modifiers.new("uplift", "NODES"); md.node_group = uplift_tree()
        res["contact"].append(cob)
        mpts, ms = wire_polyline(off, s_m, 1.0, "messenger")
        mob = tube_mesh(f"Messenger{tr}", mpts, 0.0065, coll, 6, cu, attr_s=ms)
        md = mob.modifiers.new("uplift", "NODES"); md.node_group = uplift_tree()
        res["messenger"].append(mob)
        # droppers every ~6.85 m (7 per span), vertical between messenger and contact wire
        dverts, dfaces = [], []
        for i in range(len(s_m) - 1):
            for k in range(1, 7):
                f = (k - 0.5 + 0.5) / 7.0
                j = int(round((i + f) * S.MAST_SPACING_VIADUCT / 0.5))
                jm = int(round((i + f) * S.MAST_SPACING_VIADUCT / 1.0))
                if j >= len(cpts) or jm >= len(mpts):
                    continue
                a = cpts[j]; b = mpts[jm]
                b = np.array([a[0], a[1], b[2]])
                ob_d = tube_between("drop", a, b, 0.0025, coll, 4)
                dverts.append(ob_d)
        if dverts:
            dj = join(dverts, f"Droppers{tr}"); coll.objects.link(dj); dj.data.materials.clear(); dj.data.materials.append(cu)
        # feeder (AF) and protection (PW) wires along the mast tops with catenary sag
        for name, lat_extra, z_off, rr in (("Feeder", 0.8, -0.1 + 0.35, 0.011), ("PW", 0.2, -1.2, 0.007)):
            pts = []
            for i in range(len(s_m) - 1):
                for k in range(12):
                    f = k / 12
                    s = s_m[i] + f * (s_m[i + 1] - s_m[i])
                    Ofs, Tfs, Nfs, Ufs = _frames_flat(np.array([s]))
                    p = Ofs[0] + Nfs[0] * (mast_lat + tr * lat_extra)
                    p[2] = A.RAIL_Z + 8.6 + z_off - 4 * 0.9 * f * (1 - f)
                    pts.append(p)
            tube_mesh(f"{name}{tr}", np.array(pts), rr, coll, 6, blk)
    return res


def build_furniture(coll, ground_fn):
    """Balises between the rails, km/hm posts on the cess/parapet, GSM-R mast beside the line."""
    from .trainparts import box, setmat, cylinder, join, tube_between
    yel = M.principled("Balise", (0.75, 0.55, 0.02), 0.0, 0.45)
    white = M.principled("PostWhite", (0.80, 0.80, 0.78), 0.0, 0.6)
    galv = M.principled("Galvanised", (0.55, 0.56, 0.57), 1.0, 0.38)
    objs = []
    # balises: groups of two, 3 m apart, every ~1.2 km on both tracks (CTCS-3 position references)
    src = setmat(box("BaliseSrc", (0.52, 0.36, 0.06), (0, 0, -0.14), coll), yel); src.hide_render = True; src.hide_viewport = True
    ss = np.concatenate([[s, s + 3.0] for s in np.arange(A.S_MIN + 300, A.TUNNEL_S, 1200.0)])
    for tr in (+1, -1):
        O, T, N, U = A.frame(ss, tr * S.TRACK_CENTRES / 2)
        objs.append(instancer(f"Balises{tr}", src, O, matrix_to_euler(T, N, U), coll=coll))
    # hectometre posts: small white plates on the parapet / cess every 100 m (left side), km posts larger
    hp = setmat(box("HmSrc", (0.06, 0.30, 0.40), (0, 0, 0.20), coll), white); hp.hide_render = True; hp.hide_viewport = True
    s_h = np.arange(np.ceil(A.S_MIN / 100) * 100, A.TUNNEL_S, 100.0)
    O, T, N, U = _frames_flat(s_h)
    base = O + N * 6.15
    base[:, 2] = np.where((s_h > A.VIADUCT[0]) & (s_h < A.VIADUCT[1]), A.RAIL_Z - GIRDER_TOP_BELOW_RAIL + 1.15, A.RAIL_Z - 0.85)
    objs.append(instancer("HmPosts", hp, base, matrix_to_euler(T, N, U), coll=coll))
    # GSM-R monopole (35 m) with antenna platform, 40 m south of the line near s = 1650
    O, T, N, U = _frames_flat(np.array([1650.0]))
    p = O[0] + N[0] * -42.0
    g = ground_fn(p[0], p[1])
    parts = [setmat(tube_between("pole", (p[0], p[1], g), (p[0], p[1], g + 36.0), 0.45, coll, 16), galv),
             setmat(tube_between("pole2", (p[0], p[1], g + 30), (p[0], p[1], g + 40.0), 0.25, coll, 12), galv)]
    for k in range(3):
        a = k * 2 * math.pi / 3
        q = np.array([p[0] + 0.9 * math.cos(a), p[1] + 0.9 * math.sin(a), g + 37.5])
        parts.append(setmat(box("ant", (0.12, 0.35, 1.6), tuple(q), coll), white))
    parts.append(setmat(cylinder("plat", 1.3, 0.12, "z", (p[0], p[1], g + 36.4), coll, 16), galv))
    ob = join(parts, "GSMR_Mast"); coll.objects.link(ob); objs.append(ob)
    return objs


def _sign_material(name, text, size_px=(256, 256), bg=(250, 250, 248), fg=(10, 10, 10), font=None, border=True, texdir="/tmp"):
    """Image-textured sign face generated with PIL (Latin digits or CJK characters)."""
    import os
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(texdir, exist_ok=True)
    path = os.path.join(texdir, f"sign_{name}.png")
    im = Image.new("RGB", size_px, bg); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font or "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", int(size_px[1] * 0.55))
    bb = d.textbbox((0, 0), text, font=f)
    d.text(((size_px[0] - (bb[2] - bb[0])) / 2 - bb[0], (size_px[1] - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=f)
    if border:
        d.rectangle([3, 3, size_px[0] - 4, size_px[1] - 4], outline=fg, width=6)
    im.save(path)
    m = bpy.data.materials.new(f"Sign_{name}"); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path)
    nt.links.new(t.outputs["Color"], p.inputs["Base Color"]); p.inputs["Roughness"].default_value = 0.5
    return m


def _plate(name, w, h, mat, coll):
    v = [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, h), (-w / 2, 0, h)]
    ob = new_mesh_object(name, np.array(v), [(0, 1, 2, 3)], coll, uvs=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]), smooth=False)
    ob.data.materials.append(mat)
    return ob


def build_signs(coll, texdir, km0=1186.0):
    """Readable kilometre posts every 1000 m and a neutral section (断 / 合 boards) on the viaduct."""
    from .trainparts import box, setmat
    objs = []
    white = M.principled("PostWhite", (0.80, 0.80, 0.78), 0.0, 0.6)
    for s in np.arange(0.0, A.TUNNEL_S, 1000.0):
        km = km0 + s / 1000.0
        mat = _sign_material(f"km{int(km)}", f"{int(km)}", (320, 200), texdir=texdir)
        O, T, N, U = _frames_flat(np.array([s])); O, T, N = O[0], T[0], N[0]
        on_viaduct = A.VIADUCT[0] < s < A.VIADUCT[1]
        base = O + N * 6.05
        base[2] = A.RAIL_Z - GIRDER_TOP_BELOW_RAIL + 1.15 if on_viaduct else A.RAIL_Z - 0.85
        post = setmat(box(f"KmPost{int(km)}", (0.10, 0.10, 0.9), tuple(base + np.array([0, 0, 0.45])), coll), white)
        pl = _plate(f"KmPlate{int(km)}", 0.48, 0.30, mat, coll)
        from mathutils import Matrix
        # plate faces the track: local x along T, local y (plate normal side) along N, z up
        pl.matrix_world = Matrix(((T[0], N[0], 0, base[0] - N[0] * 0.06), (T[1], N[1], 0, base[1] - N[1] * 0.06), (0, 0, 1, base[2] + 0.62), (0, 0, 0, 1)))
        objs += [post, pl]
    # neutral section on the viaduct: 断 board 200 m before, 合 board after (on the masts of the left track)
    cjk = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    for s, txt in ((1720.0, "断"), (2010.0, "合")):
        mat = _sign_material("ns_" + ("off" if txt == "断" else "on"), txt, (256, 256), bg=(250, 250, 250), fg=(12, 12, 12), font=cjk)
        O, T, N, U = _frames_flat(np.array([s])); O, T, N = O[0], T[0], N[0]
        p = O + N * 5.35; p[2] = A.RAIL_Z + 4.2
        from mathutils import Matrix
        pl = _plate(f"NeutralBoard_{txt}", 0.60, 0.60, mat, coll)
        pl.matrix_world = Matrix(((-N[0], T[0], 0, p[0]), (-N[1], T[1], 0, p[1]), (0, 0, 1, p[2]), (0, 0, 0, 1)))
        objs.append(pl)
    return objs


# ------------------------------------------------------------------ cutting details (v3)
LATTICE_PITCH = 5.0          # diamond frame lines s +/- lat = 5 k (same layout as the old painted lattice)


def on_lattice(s, lat, half_w=0.45):
    """True where (s, lat) lies on a lattice beam (used to keep grass tufts off the concrete)."""
    out = np.zeros(np.shape(s), bool)
    for f in (1.0, -1.0):
        c = s + f * lat
        d = np.abs(c - LATTICE_PITCH * np.round(c / LATTICE_PITCH)) / math.sqrt(2)
        out |= d < half_w
    return out


def _cut_points(s, lat):
    from . import world as W
    th = A.heading(s); x0, y0 = A.xy(s)
    x = x0 - np.sin(th) * lat; y = y0 + np.cos(th) * lat
    nat = W.natural_ground(x, y)
    g = W.formation(s, lat, nat)
    return x, y, g, nat


def build_cutting_details(coll, tex):
    """Real geometry for the cutting (S7 / S9 are shot inside it):
    - precast concrete diamond lattice frames on the cut slopes: 0.40 m wide beams standing 0.15 m proud of the
      soil (embedded 0.15 m), in 5 m diamonds, up to 15 m above rail level, so the low sun throws real shadows;
    - concrete-lined side drains (0.8 x 0.5 m U-channels) at the toe of both slopes;
    - a crushed-stone cess between the slab-track base and the drain."""
    con = concrete("LatticeConcrete", tex.get("concrete"), (0.86, 0.85, 0.81), 0.5)
    s0, s1 = A.VIADUCT[1], A.TUNNEL_S - 16.0
    w, h_top, h_bot = 0.40, 0.15, -0.15
    V, F, UV = [], [], []
    for side in (1.0, -1.0):
        for f in (1.0, -1.0):
            cs = np.arange(LATTICE_PITCH * math.floor((s0 - 40) / LATTICE_PITCH), s1 + 40, LATTICE_PITCH)
            for c in cs:
                lat = side * np.arange(7.25, 32.0, 0.4)
                s = c - f * lat
                x, y, g, nat = _cut_points(s, lat)
                ok = (s > s0) & (s < s1) & (nat > g + 0.3) & (g < A.RAIL_Z + 15.0)
                if ok.sum() < 3:
                    continue
                # contiguous runs
                idx = np.flatnonzero(ok)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                for r in runs:
                    if len(r) < 3:
                        continue
                    ss, ll = s[r], lat[r]
                    e = np.array([1.0, f * side]) / math.sqrt(2)          # across the beam, in (s, |lat|) space
                    rows = []
                    for k in (-1, 1):
                        sk = ss + k * w / 2 * e[0]; lk = ll + k * w / 2 * e[1] * side
                        xk, yk, gk, _ = _cut_points(sk, lk)
                        rows.append((xk, yk, gk))
                    m = len(r)
                    base = len(V)
                    for i in range(m):
                        for (xk, yk, gk) in rows:
                            V.append((xk[i], yk[i], gk[i] + h_top))
                        for (xk, yk, gk) in rows:
                            V.append((xk[i], yk[i], gk[i] + h_bot))
                    # vertex layout per i: 0 L_top, 1 R_top, 2 L_bot, 3 R_bot
                    along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ss), np.diff(ll)) * 1.0)])
                    for i in range(m - 1):
                        a, b = base + 4 * i, base + 4 * (i + 1)
                        for quad, u0, u1 in (((a, a + 1, b + 1, b), 0.0, w), ((a + 2, a, b, b + 2), -0.3, 0.0),
                                             ((a + 1, a + 3, b + 3, b + 1), w, w + 0.3)):
                            F.append(quad)
                            UV += [(u0, along[i]), (u1, along[i]), (u1, along[i + 1]), (u0, along[i + 1])]
                    for (a0) in (base, base + 4 * (m - 1)):
                        F.append((a0, a0 + 1, a0 + 3, a0 + 2)); UV += [(0, 0), (w, 0), (w, 0.3), (0, 0.3)]
    ob = new_mesh_object("SlopeLattice", np.array(V), F, coll, smooth=False, uvs=np.array(UV))
    ob.data.materials.append(con)
    log_n = len(F)
    # side drains + cess (sweeps relative to rail level on the centre line)
    from . import world as W
    ss = np.arange(s0, s1, 2.0)
    _, _, _, nat = _cut_points(ss, np.full(len(ss), 7.5))
    cut = ss[nat > A.RAIL_Z - 0.85 + 0.4]
    objs = [ob]
    if len(cut) > 2:
        c0, c1 = float(cut.min()) + 6.0, float(min(cut.max(), A.TUNNEL_S - 14.5))
        drain = concrete("DrainConcrete", tex.get("concrete"), (0.80, 0.79, 0.75), 0.5)
        cess = concrete("CessGravel", tex.get("gravel"), (0.85, 0.83, 0.80), 0.6, streaks=False)
        for side in (1.0, -1.0):
            prof = [(5.75, -0.80), (5.85, -0.80), (5.85, -1.28), (6.65, -1.28), (6.65, -0.80), (6.75, -0.80),
                    (6.75, -1.40), (5.75, -1.40)]
            prof = [(side * a, b) for a, b in prof]
            if side < 0:
                prof = prof[::-1]
            objs.append(sweep(f"SideDrain{int(side)}", prof, c0, c1, 2.0, coll=coll, mat=drain, closed=True))
            off = A.track_offset() if side > 0 else -A.track_offset()
            inner = abs(off) + 1.50
            cp = [(side * inner, -0.84), (side * 5.75, -0.82), (side * 5.75, -0.95), (side * inner, -0.97)]
            if side < 0:
                cp = cp[::-1]
            objs.append(sweep(f"Cess{int(side)}", cp, c0, c1, 2.0, coll=coll, mat=cess, closed=True, uv_scale=1.0))
    print(f"[build]   cutting: lattice {log_n} faces, drains + cess", flush=True)
    return objs
