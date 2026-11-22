"""Mechanical parts and interiors of the CR450AF (bpy).  Units m, z = 0 top of rail."""
import math

import bpy
import numpy as np

from . import specs as S
from . import trainshape as TS
from . import materials as M
from .bl import new_mesh_object, grid_faces


# ------------------------------------------------------------------ primitives
def revolve(name, prof, axis="y", segs=48, coll=None, cap=False):
    """Revolve a (r, a) profile (a along the axis) around the axis through the origin."""
    prof = np.asarray(prof, float)
    verts = []
    for k in range(segs):
        th = 2 * math.pi * k / segs
        c, s = math.cos(th), math.sin(th)
        for r, a in prof:
            if axis == "y":
                verts.append((r * c, a, r * s))
            elif axis == "x":
                verts.append((a, r * c, r * s))
            else:
                verts.append((r * c, r * s, a))
    n = len(prof)
    faces = []
    for k in range(segs):
        k2 = (k + 1) % segs
        for i in range(n - 1):
            faces.append((k * n + i, k * n + i + 1, k2 * n + i + 1, k2 * n + i))
    return new_mesh_object(name, np.array(verts), faces, coll)


def box(name, size, loc=(0, 0, 0), coll=None, bevel=0.0):
    sx, sy, sz = (v / 2 for v in size)
    v = np.array([(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
                  (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]) + np.array(loc)
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    ob = new_mesh_object(name, v, f, coll, smooth=False)
    if bevel:
        md = ob.modifiers.new("bevel", "BEVEL"); md.width = bevel; md.segments = 2
    return ob


def cylinder(name, r, length, axis="y", loc=(0, 0, 0), coll=None, segs=24):
    prof = [(0.0, -length / 2), (r, -length / 2), (r, length / 2), (0.0, length / 2)]
    ob = revolve(name, prof, axis, segs, coll)
    ob.location = loc
    return ob


def tube_between(name, p0, p1, r, coll=None, segs=12):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    d = p1 - p0; L = np.linalg.norm(d)
    ob = cylinder(name, r, L, "z", coll=coll, segs=segs)
    from mathutils import Vector
    ob.location = tuple((p0 + p1) / 2)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector(tuple(d / L)).to_track_quat("Z", "Y")
    return ob


def join(objs, name):
    """Join objects (applying their own loc/rot/scale + modifiers) into one new mesh object (not linked)."""
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    mats, matidx = [], {}
    need_eval = any(len(o.modifiers) for o in objs)
    dg = bpy.context.evaluated_depsgraph_get() if need_eval else None
    for o in objs:
        if o.modifiers:
            oe = o.evaluated_get(dg); m = oe.to_mesh().copy(); oe.to_mesh_clear()
        else:
            m = o.data.copy()
        m.transform(o.matrix_basis)
        remap = []
        for ms in o.data.materials:
            if ms.name not in matidx:
                matidx[ms.name] = len(mats); mats.append(ms)
            remap.append(matidx[ms.name])
        if remap:
            idx = np.zeros(len(m.polygons), dtype=np.int32)
            m.polygons.foreach_get("material_index", idx)
            idx = np.array(remap, dtype=np.int32)[np.clip(idx, 0, len(remap) - 1)]
            m.polygons.foreach_set("material_index", idx)
        bm.from_mesh(m)
        bpy.data.meshes.remove(m)
    bm.to_mesh(me); bm.free()
    for ms in mats:
        me.materials.append(ms)
    for o in objs:
        d = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if d.users == 0:
            bpy.data.meshes.remove(d)
    return bpy.data.objects.new(name, me)


def setmat(ob, mat):
    ob.data.materials.clear(); ob.data.materials.append(mat)
    return ob


# ------------------------------------------------------------------ wheelset & bogie
def wheel_profile():
    """Half profile of a 920 mm wheel (r, y) from axle hub to tread/flange; y measured from wheel inner face."""
    R = S.WHEEL_DIAMETER / 2
    return [(0.0, 0.035), (0.10, 0.035), (0.10, -0.06), (0.13, -0.06), (0.16, 0.0), (0.30, 0.02), (0.37, 0.0),
            (R - 0.05, 0.0), (R + 0.028, 0.0), (R + 0.030, 0.012), (R + 0.012, 0.030),   # flange (inside)
            (R, 0.040), (R - 0.004, 0.135),                                          # 1:40 tread
            (R - 0.03, 0.135), (R - 0.045, 0.125), (0.33, 0.11), (0.17, 0.11), (0.12, 0.17), (0.0, 0.17)]


def build_wheelset(name, coll, motor=False):
    bb = 1.353 / 2
    mats = dict(wheel=M.steel("WheelSteel", (0.42, 0.40, 0.38), 0.35), axle=M.steel("Axle", (0.20, 0.19, 0.18), 0.5),
                disc=M.steel("BrakeDisc", (0.35, 0.33, 0.31), 0.3))
    parts = []
    prof = wheel_profile()
    for side in (1, -1):
        p = [(r, side * (bb + a)) for r, a in prof]
        if side < 0:
            p = p[::-1]
        w = revolve(f"{name}_w{side}", p, "y", 64, coll)
        setmat(w, mats["wheel"]); parts.append(w)
        # wheel-mounted brake disc ring with bolt circle (visible rotation cue)
        d = revolve(f"{name}_wd{side}", [(0.20, side * (bb - 0.035)), (0.36, side * (bb - 0.035)), (0.36, side * (bb - 0.06)), (0.20, side * (bb - 0.06))], "y", 48, coll)
        setmat(d, mats["disc"]); parts.append(d)
        for k in range(8):
            th = 2 * math.pi * k / 8
            bolt = cylinder(f"{name}_b{side}{k}", 0.018, 0.03, "y", (0.27 * math.cos(th), side * (bb - 0.075), 0.27 * math.sin(th)), coll, 8)
            setmat(bolt, mats["axle"]); parts.append(bolt)
    ax = cylinder(f"{name}_axle", 0.095, 2.06, "y", (0, 0, 0), coll, 24); setmat(ax, mats["axle"]); parts.append(ax)
    if not motor:
        for y in (-0.33, 0.0, 0.33):
            d = cylinder(f"{name}_ad{y}", 0.32, 0.11, "y", (0, y, 0), coll, 40); setmat(d, mats["disc"]); parts.append(d)
    else:
        g = box(f"{name}_gear", (0.55, 0.28, 0.62), (0, 0.25, 0.05), coll); setmat(g, M.principled("GearBox", (0.06, 0.06, 0.065), 0.3, 0.6)); parts.append(g)
    ob = join(parts, name)
    coll.objects.link(ob)
    return ob


def build_bogie(name, coll, motor=False):
    """Bogie frame (no wheelsets).  Origin at the pivot centre, z = 0 top of rail."""
    R = S.WHEEL_DIAMETER / 2
    wb = S.BOGIE_WHEELBASE / 2
    frame_m = M.principled("BogieFrame", (0.045, 0.047, 0.05), 0.2, 0.55)
    rubber = M.principled("Rubber", (0.02, 0.02, 0.02), 0.0, 0.7)
    yel = M.principled("DamperYellow", (0.6, 0.42, 0.05), 0.0, 0.5)
    parts = []
    for side in (1, -1):
        y = side * 1.02
        # side beam: ends over axle boxes, dropped centre
        for x0, x1, zc, h in ((-wb - 0.35, -0.55, R + 0.18, 0.24), (-0.55, 0.55, R + 0.02, 0.30), (0.55, wb + 0.35, R + 0.18, 0.24)):
            b = box(f"{name}_sb", (x1 - x0, 0.16, h), ((x0 + x1) / 2, y, zc), coll, 0.02); setmat(b, frame_m); parts.append(b)
        for x in (-wb, wb):
            ab = box(f"{name}_ab", (0.34, 0.24, 0.26), (x, side * 1.05, R), coll, 0.02); setmat(ab, frame_m); parts.append(ab)
            for dx in (-0.21, 0.21):   # primary coil springs
                sp = revolve(f"{name}_sp", [(0.0, -0.12), (0.085, -0.12), (0.085, 0.12), (0.0, 0.12)], "z", 16, coll)
                sp.location = (x + dx, side * 1.05, R + 0.18); setmat(sp, M.principled("Spring", (0.15, 0.03, 0.02), 0.6, 0.5)); parts.append(sp)
        # air spring
        a = revolve(f"{name}_air", [(0.0, -0.13), (0.30, -0.13), (0.36, 0.0), (0.30, 0.13), (0.0, 0.13)], "z", 32, coll)
        a.location = (0, side * 1.02, R + 0.32); setmat(a, rubber); parts.append(a)
        # yaw damper
        yd = tube_between(f"{name}_yd", (-1.2, side * 1.30, R + 0.25), (0.6, side * 1.30, R + 0.30), 0.045, coll)
        setmat(yd, yel); parts.append(yd)
        vd = tube_between(f"{name}_vd", (wb - 0.2, side * 1.25, R - 0.05), (wb - 0.05, side * 1.25, R + 0.35), 0.035, coll)
        setmat(vd, frame_m); parts.append(vd)
    for x in (-0.35, 0.35):
        cb = box(f"{name}_cb", (0.20, 2.04, 0.22), (x, 0, R + 0.02), coll, 0.02); setmat(cb, frame_m); parts.append(cb)
    if motor:
        for x in (-0.80, 0.80):
            mo = cylinder(f"{name}_mot", 0.26, 0.70, "y", (x * 0.85, -0.15, R + 0.05), coll, 24)
            setmat(mo, M.principled("Motor", (0.05, 0.05, 0.055), 0.4, 0.5)); parts.append(mo)
    else:
        for x in (-wb, wb):
            for side in (1, -1):
                cal = box(f"{name}_cal", (0.25, 0.10, 0.30), (x + 0.0, side * 0.33, R + 0.30), coll); setmat(cal, frame_m); parts.append(cal)
    ob = join(parts, name)
    coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------ pantograph
def panto_knee(head_h, base=(-0.75, 0.45), l1=1.55, l2=1.70):
    """Two-link IK in the x-z plane: hinge at base, head at (0, head_h). Knee leads (+x)."""
    bx, bz = base
    hx, hz = 0.0, head_h
    dx, dz = hx - bx, hz - bz
    d = math.hypot(dx, dz)
    d = min(d, l1 + l2 - 1e-4)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0))
    mx, mz = bx + a * dx / d, bz + a * dz / d
    # pick knee on the +x side
    k1 = (mx + h * dz / d, mz - h * dx / d); k2 = (mx - h * dz / d, mz + h * dx / d)
    return k1 if k1[0] > k2[0] else k2


def build_pantograph(name, coll, head_h=1.00):
    """Single-arm pantograph; origin on the roof at mounting centre.  head_h = carbon-strip height above origin.

    Returns (base_obj, arm_rig) where arm parts are separate objects so that they can be re-posed per frame.
    """
    alu = M.principled("PantoAlu", (0.55, 0.56, 0.58), 1.0, 0.35)
    ins = M.principled("Insulator", (0.20, 0.06, 0.05), 0.0, 0.25, coat=0.6)
    dark = M.principled("PantoDark", (0.05, 0.05, 0.05), 0.5, 0.5)
    carbon = M.principled("Carbon", (0.03, 0.03, 0.03), 0.2, 0.35)
    parts = []
    for x in (-0.55, 0.45):
        for y in (-0.38, 0.38):
            prof = [(0.0, 0.0), (0.05, 0.0)]
            for k in range(6):
                z0 = 0.04 + k * 0.05
                prof += [(0.05, z0), (0.085, z0 + 0.012), (0.05, z0 + 0.03)]
            prof += [(0.05, 0.36), (0.0, 0.36)]
            i = revolve(f"{name}_ins", prof, "z", 16, coll); i.location = (x, y, 0.0); setmat(i, ins); parts.append(i)
    fr = box(f"{name}_frame", (1.25, 0.95, 0.08), (-0.05, 0, 0.40), coll, 0.01); setmat(fr, alu); parts.append(fr)
    fr2 = box(f"{name}_frameh", (1.0, 0.75, 0.08), (-0.05, 0, 0.40), coll); setmat(fr2, alu)
    # cut the frame: keep it as a solid plate (reads as frame at distance)
    bpy.data.objects.remove(fr2, do_unlink=True)
    hinge = cylinder(f"{name}_hinge", 0.06, 0.70, "y", (-0.75, 0, 0.47), coll, 16); setmat(hinge, alu); parts.append(hinge)
    damper = tube_between(f"{name}_dmp", (-0.55, 0.25, 0.45), (-0.2, 0.25, 0.75), 0.03, coll); setmat(damper, dark); parts.append(damper)
    base = join(parts, name + "_base"); coll.objects.link(base)
    rig = dict(alu=alu, carbon=carbon, dark=dark)
    return base, rig


def pose_pantograph(name, coll, rig, head_h, parent):
    """Single-arm pantograph posed for a head height above the mount (m).

    Lower arm: tapered A-frame from the base hinge to the knee; upper arm: slim tube from the knee back to the
    head; balance (guide) rod; pneumatic damper; sprung pan head 1950 mm wide with two carbon strips and horns.
    """
    for o in list(coll.objects):
        if o.name.startswith(name + "_arm"):
            bpy.data.objects.remove(o, do_unlink=True)
    hx, hz = 0.0, head_h - 0.30
    base = (-0.85, 0.47)
    kx, kz = panto_knee(hz, base=base, l1=1.62, l2=1.74)
    parts = []
    # lower arm A-frame: two tubes from the hinge ends converging at the knee, with two cross braces
    for y in (-0.34, 0.34):
        parts.append(setmat(tube_between(f"{name}_l", (base[0], y, base[1]), (kx, y * 0.18, kz), 0.042, coll, 10), rig["alu"]))
    for f in (0.35, 0.7):
        ax = base[0] + (kx - base[0]) * f; az = base[1] + (kz - base[1]) * f
        w = 0.34 * (1 - f) + 0.06 * f
        parts.append(setmat(tube_between(f"{name}_lb", (ax, -w, az), (ax, w, az), 0.018, coll, 6), rig["alu"]))
    # knee joint
    parts.append(setmat(cylinder(f"{name}_kn", 0.05, 0.16, "y", (kx, 0, kz), coll, 12), rig["alu"]))
    # upper arm: single slim tube knee -> head frame, plus guide rod below it
    parts.append(setmat(tube_between(f"{name}_u", (kx, 0, kz), (hx, 0, hz), 0.032, coll, 10), rig["alu"]))
    parts.append(setmat(tube_between(f"{name}_gr", (base[0] + 0.22, 0, base[1] + 0.02), (kx - 0.10, 0, kz - 0.12), 0.013, coll, 6), rig["dark"]))
    # damper (pneumatic cylinder) on the base pushing the lower arm
    parts.append(setmat(tube_between(f"{name}_dmp", (base[0] + 0.55, 0.20, base[1]), (base[0] + (kx - base[0]) * 0.25, 0.12, base[1] + (kz - base[1]) * 0.25), 0.035, coll, 8), rig["dark"]))
    # pan head: frame, springs, two carbon strips across the track, horns
    parts.append(setmat(box(f"{name}_hf", (0.46, 0.08, 0.05), (hx, 0, hz + 0.04), coll), rig["alu"]))
    for side in (1, -1):
        parts.append(setmat(tube_between(f"{name}_hs", (hx - 0.18, side * 0.42, hz + 0.05), (hx + 0.18, side * 0.42, hz + 0.05), 0.02, coll, 6), rig["alu"]))
        for x in (-0.18, 0.18):
            parts.append(setmat(tube_between(f"{name}_sp", (x, side * 0.42, hz + 0.05), (x, side * 0.42, head_h - 0.04), 0.018, coll, 6), rig["dark"]))
    for x in (-0.18, 0.18):
        parts.append(setmat(box(f"{name}_cs", (0.055, 1.30, 0.032), (x, 0, head_h - 0.016), coll), rig["carbon"]))
        parts.append(setmat(box(f"{name}_csb", (0.07, 1.36, 0.018), (x, 0, head_h - 0.042), coll), rig["alu"]))
        for side in (1, -1):
            pts = [(x, side * 0.68, head_h - 0.03), (x, side * 0.82, head_h - 0.06), (x, side * 0.95, head_h - 0.16), (x, side * 0.97, head_h - 0.26)]
            for a, b in zip(pts[:-1], pts[1:]):
                parts.append(setmat(tube_between(f"{name}_hn", a, b, 0.016, coll, 6), rig["alu"]))
    arm = join(parts, name + "_arm")
    coll.objects.link(arm)
    arm.parent = parent
    return arm


def build_roof_equipment(name, coll, car_length, panto=True):
    """Pantograph-well side shrouds and the roof HV cable on post insulators (car-local, roof at 3.86 m)."""
    shroud = M.principled("RoofShroud", (0.56, 0.57, 0.59), 0.85, 0.35, coat=0.4)
    ins = M.principled("Insulator", (0.20, 0.06, 0.05), 0.0, 0.25, coat=0.6)
    cable = M.principled("HVCable", (0.03, 0.03, 0.03), 0.0, 0.45)
    parts = []
    if panto:
        for side in (1, -1):
            pts = [(-2.4, 0.0), (-1.6, 0.30), (1.6, 0.30), (2.4, 0.0)]
            for (x0, h0), (x1, h1) in zip(pts[:-1], pts[1:]):
                b = box(f"{name}_sh", (x1 - x0, 0.05, max(h0, h1)), ((x0 + x1) / 2, side * 0.78, 3.80 + max(h0, h1) / 2), coll)
                setmat(b, shroud); parts.append(b)
        # HV cable from the pantograph to the car end (roof bus), on post insulators every 3 m
        for x in np.arange(-1.6, -car_length / 2 + 1.0, -3.0):
            prof = [(0.0, 0.0), (0.045, 0.0)]
            for k in range(4):
                z0 = 0.03 + k * 0.04
                prof += [(0.045, z0), (0.07, z0 + 0.01), (0.045, z0 + 0.025)]
            prof += [(0.045, 0.2), (0.0, 0.2)]
            i = revolve(f"{name}_pi", prof, "z", 12, coll); i.location = (x, 0.35, 3.84); setmat(i, ins); parts.append(i)
        parts.append(setmat(tube_between(f"{name}_cab", (-0.6, 0.35, 4.06), (-car_length / 2 + 1.0, 0.35, 4.06), 0.028, coll, 8), cable))
    if not parts:
        return None
    ob = join(parts, name); coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------ gangway, deflector, wiper
def build_gangway(name, coll, length=0.62):
    hs = TS.body_section()
    ring = TS.full_ring(hs)
    c = np.array([0.0, 2.0])
    ring = (ring - c) * np.array([0.985, 0.985]) + c
    n = len(ring)
    xs = np.linspace(-length / 2, length / 2, 9)
    verts = []
    for i, x in enumerate(xs):
        inset = 0.012 * (1 - abs(2 * i / (len(xs) - 1) - 1))
        r2 = (ring - c) * (1 - inset) + c
        verts.extend((x, p[0], p[1]) for p in r2)
    faces = grid_faces(len(xs), n, closed=True)
    ob = new_mesh_object(name, np.array(verts), faces, coll)
    setmat(ob, M.gangway_material())
    return ob


def build_deflector(name, coll, X_origin):
    """Obstacle deflector / lower front spoiler under the bill (dark grey)."""
    prof = []  # wedge in plan
    pts = []
    for X, hw in ((1.05, 0.62), (1.5, 0.85), (2.2, 1.12), (3.0, 1.36), (4.2, 1.50)):
        pts.append((X, hw))
    verts = []
    for X, hw in pts:
        for y in (-hw, hw):
            verts.append((X_origin - X, y, 0.16)); verts.append((X_origin - X, y, 0.24))
    faces = []
    n = len(pts)
    for i in range(n - 1):
        a = 4 * i; b = 4 * (i + 1)
        faces += [(a, b, b + 2, a + 2), (a + 1, a + 3, b + 3, b + 1), (a, a + 1, b + 1, b), (a + 2, b + 2, b + 3, a + 3)]
    faces += [(0, 2, 3, 1)]
    ob = new_mesh_object(name, np.array(verts), faces, coll, smooth=False)
    setmat(ob, M.principled("Deflector", (0.55, 0.56, 0.58), 0.85, 0.35))
    md = ob.modifiers.new("bevel", "BEVEL"); md.width = 0.03; md.segments = 2
    return ob


def build_emblem(name, coll, X_origin):
    """Gold CR emblem (ring + rail-profile glyph) on the black face above the bill."""
    gold = principled_gold()
    X, z = 1.30, 1.64
    hs = TS.half_section(X)
    parts = []
    ring = revolve(name + "_r", [(0.115, -0.006), (0.135, -0.006), (0.135, 0.006), (0.115, 0.006)], "x", 40, coll)
    setmat(ring, gold); parts.append(ring)
    for (sz, loc) in (((0.012, 0.13, 0.025), (0, 0, 0.055)), ((0.012, 0.03, 0.10), (0, 0, 0.0)), ((0.012, 0.10, 0.022), (0, 0, -0.055))):
        b = box(name + "_g", sz, loc, coll); setmat(b, gold); parts.append(b)
    ob = join(parts, name); coll.objects.link(ob)
    ob.location = (X_origin - X + 0.01, 0, z)
    ob.rotation_euler = (0, math.radians(-38), 0)
    return ob


def principled_gold():
    return M.principled("EmblemGold", (0.78, 0.55, 0.20), 1.0, 0.3)


def build_wiper(name, coll, X_origin):
    """Single pantograph wiper parked along the lower windscreen edge (follows the nose surface)."""
    X = 2.30
    hs = TS.half_section(X)
    top = hs[-1]
    pts = []
    for y in np.linspace(-0.05, 0.85, 12):
        # find z on the section for lateral y (upper part)
        up = hs[TS.CREASE_IDX:]
        k = np.argmin(np.abs(up[:, 0] - y))
        pts.append((X_origin - X - 0.02 * (y + 0.05), -y, up[k, 1] + 0.025))
    objs = []
    dark = M.principled("Wiper", (0.015, 0.015, 0.015), 0.3, 0.4)
    for a, b in zip(pts[:-1], pts[1:]):
        objs.append(setmat(tube_between(name + "_s", a, b, 0.012, coll, 8), dark))
    ob = join(objs, name); coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------ interiors
def build_seat_row(name, coll):
    """One 2+3 second-class row (grey-blue seats) as a single mesh, centred at x=0."""
    fab = M.principled("SeatFabric", (0.10, 0.13, 0.20), 0.0, 0.8)
    head = M.principled("Headrest", (0.75, 0.74, 0.70), 0.0, 0.7)
    parts = []
    ys = [1.18, 0.70, -0.30, -0.78, -1.26]
    for y in ys:
        parts.append(setmat(box(name + "_c", (0.48, 0.44, 0.12), (0.05, y, 1.20 + 0.44), coll, 0.03), fab))
        parts.append(setmat(box(name + "_b", (0.10, 0.44, 0.70), (-0.18, y, 1.20 + 0.85), coll, 0.03), fab))
        parts.append(setmat(box(name + "_h", (0.06, 0.40, 0.12), (-0.21, y, 1.20 + 1.17), coll, 0.02), head))
    ob = join(parts, name); coll.objects.link(ob)
    return ob


def build_car_interior(name, coll, length, seat_row_mesh, rows_x):
    """Floor, ceiling with light strips, luggage racks, seats.  x along car centred."""
    floor = M.principled("Floor", (0.12, 0.11, 0.10), 0.0, 0.6)
    ceil = M.principled("Ceiling", (0.70, 0.70, 0.68), 0.0, 0.5)
    light = M.principled("CeilLight", (1, 1, 1), 0.0, 0.5, emission=(1.0, 0.95, 0.88), estrength=2.0)
    rack = M.principled("Rack", (0.6, 0.6, 0.62), 1.0, 0.3)
    parts = [setmat(box(name + "_floor", (length, 3.0, 0.04), (0, 0, 1.20), coll), floor),
             setmat(box(name + "_ceil", (length, 2.4, 0.04), (0, 0, 3.32), coll), ceil),
             setmat(box(name + "_lt", (length, 0.12, 0.02), (0, 0, 3.29), coll), light)]
    for side in (1, -1):
        parts.append(setmat(box(name + "_rack", (length, 0.40, 0.03), (0, side * 1.30, 2.95), coll), rack))
    base = join(parts, name); coll.objects.link(base)
    for i, x in enumerate(rows_x):
        o = bpy.data.objects.new(f"{name}_row{i}", seat_row_mesh.data)
        o.location = (x, 0, 0); o.parent = base
        coll.objects.link(o)
    return base


def build_cab(name, coll, X_origin):
    dark = M.principled("CabDash", (0.03, 0.03, 0.035), 0.0, 0.6)
    seat = M.principled("CabSeat", (0.05, 0.05, 0.06), 0.0, 0.7)
    screen = M.principled("CabScreen", (0.02, 0.05, 0.08), 0.0, 0.2, emission=(0.2, 0.5, 0.9), estrength=0.6)
    # driver's console: sloping desk from the windscreen base back to the driver, instruments on top
    desk_v = []
    for X, z in ((4.10, 2.24), (4.60, 2.34), (5.15, 2.46), (5.55, 2.52)):
        for y in (-1.15, 1.15):
            desk_v.append((X_origin - X, y, z)); desk_v.append((X_origin - X, y, 1.60))
    desk_f = []
    for k in range(3):
        a = 4 * k; b = 4 * (k + 1)
        desk_f += [(a, a + 2, b + 2, b), (a + 1, b + 1, b + 3, a + 3), (a, b, b + 1, a + 1), (a + 2, a + 3, b + 3, b + 2)]
    desk = new_mesh_object(name + "_desk", np.array(desk_v), desk_f, coll, smooth=False)
    desk.data.materials.append(dark)
    parts = [desk,
             setmat(box(name + "_scr1", (0.22, 0.34, 0.015), (X_origin - 5.05, 0.30, 2.455), coll), screen),
             setmat(box(name + "_scr2", (0.22, 0.34, 0.015), (X_origin - 5.05, -0.30, 2.455), coll), screen),
             setmat(box(name + "_scr3", (0.18, 0.26, 0.015), (X_origin - 5.05, 0.0, 2.455), coll), screen),
             setmat(box(name + "_ctl", (0.20, 0.26, 0.08), (X_origin - 5.45, -0.55, 2.55), coll, 0.02), seat),
             setmat(box(name + "_scr1", (0.20, 0.30, 0.02), (X_origin - 5.75, 0.28, 2.18), coll), screen),
             setmat(box(name + "_scr2", (0.20, 0.30, 0.02), (X_origin - 5.75, -0.28, 2.18), coll), screen),
             setmat(box(name + "_seat", (0.55, 0.55, 0.9), (X_origin - 6.3, 0.0, 1.85), coll, 0.05), seat),
             setmat(box(name + "_wall", (0.06, 3.2, 2.3), (X_origin - 7.6, 0, 2.35), coll), dark)]
    ob = join(parts, name); coll.objects.link(ob)
    return ob
