"""Rural Guangxi houses: a modular kit of five building types, instanced into villages and hamlets.

Reference (self-built rural houses of the Guilin / Liuzhou countryside): 2-4 storey reinforced-concrete frame
houses with a tiled front facade and bare red-brick side walls, front balconies with balustrades or tiled
parapets, flat roofs with a stair house, a stainless solar water heater and a black water tank; unfinished
houses with an exposed concrete frame, brick infill and rebar stubs on the roof; older 1-2 storey grey-brick
and rendered houses with grey clay pantile gable roofs, a walled front yard and a lean-to shed; concrete
drying yards in front of the houses.

All geometry is real (window openings are holes in the wall with reveals, sills and lintels, frames and
glass set 10 cm back), so the low morning sun casts real shadows in every reveal.  Textures (Poly Haven,
CC0) are box-projected in object space at their real-world size, so they stay put on instanced copies and
never show UV seams; every instance gets its own colour/value variation and rain/splash grime.
"""
import math
import os

import bpy
import numpy as np

H_STOREY = 3.3
T_WALL = 0.24


# ------------------------------------------------------------------ mesh builder
class MB:
    """Accumulates polygons with material slots, then makes one mesh."""

    def __init__(self):
        self.v = []
        self.f = []
        self.m = []

    def poly(self, pts, mat):
        i0 = len(self.v)
        self.v.extend([tuple(map(float, p)) for p in pts])
        self.f.append(list(range(i0, i0 + len(pts))))
        self.m.append(mat)

    def hexa(self, c, mat, skip=()):
        """8 corners c[0..7] (bottom 0-3 ccw seen from above, top 4-7), 6 quads (optionally skip some)."""
        faces = {"bottom": (0, 3, 2, 1), "top": (4, 5, 6, 7), "s0": (0, 1, 5, 4), "s1": (1, 2, 6, 5),
                 "s2": (2, 3, 7, 6), "s3": (3, 0, 4, 7)}
        for k, q in faces.items():
            if k not in skip:
                self.poly([c[i] for i in q], mat)

    def box(self, x0, x1, y0, y1, z0, z1, mat, skip=()):
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        self.hexa(c, mat, skip)

    def cyl(self, p0, p1, r, mat, segs=10, caps=True):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        a = p1 - p0; a /= np.linalg.norm(a)
        u = np.cross(a, (0, 0, 1) if abs(a[2]) < 0.9 else (1, 0, 0)); u /= np.linalg.norm(u)
        v = np.cross(a, u)
        ang = np.linspace(0, 2 * math.pi, segs, endpoint=False)
        ring = [r * (math.cos(t) * u + math.sin(t) * v) for t in ang]
        for k in range(segs):
            q = [p0 + ring[k], p0 + ring[(k + 1) % segs], p1 + ring[(k + 1) % segs], p1 + ring[k]]
            self.poly(q, mat)
        if caps:
            self.poly([p0 + ring[k] for k in reversed(range(segs))], mat)
            self.poly([p1 + ring[k] for k in range(segs)], mat)

    def make(self, name, mats, recalc=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        for mt in mats:
            me.materials.append(mt)
        me.polygons.foreach_set("material_index", np.asarray(self.m, dtype=np.int32))
        me.validate(clean_customdata=False)
        if recalc:
            import bmesh
            bm = bmesh.new(); bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me); bm.free()
        me.update()
        return me


# ------------------------------------------------------------------ walls with real openings
class Wall:
    """A planar wall: outer face starts at P0, runs along U for `length`, outward normal N, thickness t inward."""

    def __init__(self, P0, U, N, length, t=T_WALL):
        self.P0 = np.asarray(P0, float); self.U = np.asarray(U, float); self.N = np.asarray(N, float)
        self.L = length; self.t = t

    def pt(self, u, z, depth):
        return self.P0 + u * self.U + np.array([0, 0, z]) - depth * self.N

    def block(self, mb, u0, u1, z0, z1, mat, d0=0.0, d1=None):
        """Solid block in wall coordinates (depth d0..d1 measured inward from the outer face; negative = proud)."""
        d1 = self.t if d1 is None else d1
        c = [self.pt(u0, z0, d0), self.pt(u1, z0, d0), self.pt(u1, z0, d1), self.pt(u0, z0, d1),
             self.pt(u0, z1, d0), self.pt(u1, z1, d0), self.pt(u1, z1, d1), self.pt(u0, z1, d1)]
        mb.hexa(c, mat)

    def plane(self, mb, u0, u1, z0, z1, depth, mat):
        mb.poly([self.pt(u0, z0, depth), self.pt(u1, z0, depth), self.pt(u1, z1, depth), self.pt(u0, z1, depth)], mat)

    def build(self, mb, z0, z1, openings, mat):
        """Solid wall between z0..z1 with rectangular openings [(u0, u1, oz0, oz1), ...] (absolute z)."""
        us = sorted({0.0, self.L, *[o[0] for o in openings], *[o[1] for o in openings]})
        zs = sorted({z0, z1, *[min(max(o[2], z0), z1) for o in openings], *[min(max(o[3], z0), z1) for o in openings]})
        for zi in range(len(zs) - 1):
            za, zb = zs[zi], zs[zi + 1]
            if zb - za < 1e-6:
                continue
            zm = (za + zb) / 2
            run = None
            for ui in range(len(us) - 1):
                ua, ub = us[ui], us[ui + 1]
                um = (ua + ub) / 2
                hole = any(o[0] < um < o[1] and o[2] < zm < o[3] for o in openings)
                if not hole:
                    run = (run[0], ub) if run else (ua, ub)
                elif run:
                    self.block(mb, run[0], run[1], za, zb, mat); run = None
            if run:
                self.block(mb, run[0], run[1], za, zb, mat)


def window_fill(mb, w, u0, u1, z0, z1, mats, kind="win", rng=None, grille=False):
    """Frame + glass (or door / shutter / dark hole) set back in an opening."""
    fr = mats["frame"]; inset = 0.10
    if kind == "hole":
        w.plane(mb, u0, u1, z0, z1, w.t * 0.98, mats["interior"])
        return
    if kind == "shutter":
        w.plane(mb, u0, u1, z0, z1, 0.14, mats["shutter"])
        w.block(mb, u0 - 0.02, u1 + 0.02, z1 - 0.32, z1, mats["shutter"], d0=-0.05, d1=0.14)     # roller box
        return
    if kind == "door":
        w.plane(mb, u0, u1, z0, z1, inset + 0.02, mats["door"])
        w.block(mb, (u0 + u1) / 2 - 0.02, (u0 + u1) / 2 + 0.02, z0, z1, mats["frame"], d0=inset - 0.02, d1=inset + 0.02)
        return
    f = 0.05
    w.block(mb, u0, u1, z0, z0 + f, fr, d0=inset - 0.02, d1=inset + 0.05)
    w.block(mb, u0, u1, z1 - f, z1, fr, d0=inset - 0.02, d1=inset + 0.05)
    w.block(mb, u0, u0 + f, z0, z1, fr, d0=inset - 0.02, d1=inset + 0.05)
    w.block(mb, u1 - f, u1, z0, z1, fr, d0=inset - 0.02, d1=inset + 0.05)
    nm = 1 if (u1 - u0) < 1.3 else 2                               # sliding sashes
    for k in range(1, nm + 1):
        um = u0 + (u1 - u0) * k / (nm + 1)
        w.block(mb, um - 0.025, um + 0.025, z0, z1, fr, d0=inset - 0.02, d1=inset + 0.05)
    w.plane(mb, u0, u1, z0, z1, inset + 0.02, mats["glass"])
    w.block(mb, u0 - 0.05, u1 + 0.05, z0 - 0.06, z0, mats["sill"], d0=-0.06, d1=0.02)             # sill
    if grille:                                                       # stainless anti-theft cage
        dd = -0.16
        for zz in (z0 + 0.05, z1 - 0.05):
            w.block(mb, u0 - 0.05, u1 + 0.05, zz - 0.015, zz + 0.015, mats["steel"], d0=dd - 0.015, d1=dd + 0.015)
        n = max(3, int((u1 - u0) / 0.13))
        for k in range(n + 1):
            uu = u0 + (u1 - u0) * k / n
            w.block(mb, uu - 0.008, uu + 0.008, z0 + 0.05, z1 - 0.05, mats["steel"], d0=dd - 0.008, d1=dd + 0.008)
        for uu in (u0 - 0.05, u1 + 0.05):
            w.block(mb, uu - 0.012, uu + 0.012, z0 + 0.04, z1 - 0.04, mats["steel"], d0=dd, d1=0.0)


def bays(L, n, ww, margin=0.0):
    """Centres of n equal bays across length L (minus margins), with window width ww."""
    c = margin + (L - 2 * margin) * (np.arange(n) + 0.5) / n
    return [(x - ww / 2, x + ww / 2) for x in c]


# ------------------------------------------------------------------ building parts
def gable_roof(mb, x0, x1, y0, y1, z_e, pitch_deg, oe, og, mats, wall_mat, ridge=True):
    """Pantile gable roof, ridge along x, eaves at y0/y1 (overhang oe), gables at x0/x1 (overhang og)."""
    tn = math.tan(math.radians(pitch_deg)); yc = (y0 + y1) / 2; th = 0.12
    zr = z_e + (yc - y0) * tn
    for (ya, yb) in ((y0 - oe, yc), (y1 + oe, yc)):
        za = z_e - (abs(ya - (y0 if ya < yc else y1))) * tn
        c = [(x0 - og, ya, za), (x1 + og, ya, za), (x1 + og, yc, zr), (x0 - og, yc, zr)]
        cu = [(x, y, z + th) for (x, y, z) in c]
        if ya > yc:
            c = [c[1], c[0], c[3], c[2]]; cu = [cu[1], cu[0], cu[3], cu[2]]
        mb.hexa(c + cu, mats["roof"])
    if ridge:
        mb.box(x0 - og - 0.05, x1 + og + 0.05, yc - 0.16, yc + 0.16, zr + th - 0.02, zr + th + 0.16, mats["ridge"])
    # gable triangles (wall material)
    for x in (x0, x1):
        xa, xb = (x, x + T_WALL) if x == x0 else (x - T_WALL, x)
        tri0 = [(xa, y0, z_e), (xa, y1, z_e), (xa, yc, zr)]
        tri1 = [(xb, y0, z_e), (xb, y1, z_e), (xb, yc, zr)]
        mb.poly(tri0[::-1], wall_mat); mb.poly(tri1, wall_mat)
        mb.poly([tri0[0], tri1[0], tri1[2], tri0[2]], wall_mat)
        mb.poly([tri0[1], tri0[2], tri1[2], tri1[1]], wall_mat)
    return zr


def solar_heater(mb, x, y, z, mats, length=1.8, rot=0.0):
    """Vacuum-tube solar water heater: stainless tank on top, 16-20 glass tubes sloping forward, steel frame."""
    ca, sa = math.cos(rot), math.sin(rot)
    def P(px, py, pz):
        return (x + ca * px - sa * py, y + sa * px + ca * py, z + pz)
    mb.cyl(P(-length / 2, 0, 1.45), P(length / 2, 0, 1.45), 0.23, mats["tank_steel"], segs=14)
    n = int(length / 0.095)
    for k in range(n):
        px = -length / 2 + 0.08 + (length - 0.16) * k / (n - 1)
        mb.cyl(P(px, -0.18, 1.32), P(px, -1.65, 0.28), 0.029, mats["tube"], segs=6, caps=False)
    for px in (-length / 2 + 0.05, length / 2 - 0.05):
        mb.cyl(P(px, 0.05, 0.0), P(px, 0.05, 1.3), 0.02, mats["steel"], segs=5)
        mb.cyl(P(px, -1.7, 0.0), P(px, -1.7, 0.25), 0.02, mats["steel"], segs=5)
        mb.cyl(P(px, 0.05, 1.25), P(px, -1.75, 0.22), 0.018, mats["steel"], segs=5)
    mb.cyl(P(-length / 2, -1.7, 0.25), P(length / 2, -1.7, 0.25), 0.02, mats["steel"], segs=5)


def water_tank(mb, x, y, z, mats, r=0.55, h=1.1):
    mb.cyl((x, y, z), (x, y, z + h), r, mats["tank_black"], segs=14)
    for k in range(3):
        a = 2 * math.pi * k / 3
        mb.box(x + 0.4 * math.cos(a) - 0.04, x + 0.4 * math.cos(a) + 0.04, y + 0.4 * math.sin(a) - 0.04,
               y + 0.4 * math.sin(a) + 0.04, z - 0.25, z, mats["concrete"])


def ac_unit(mb, w, u, z, mats):
    w.block(mb, u - 0.4, u + 0.4, z, z + 0.55, mats["ac"], d0=-0.33, d1=-0.03)
    w.block(mb, u - 0.38, u - 0.34, z - 0.05, z, mats["steel"], d0=-0.33, d1=0.0)
    w.block(mb, u + 0.34, u + 0.38, z - 0.05, z, mats["steel"], d0=-0.33, d1=0.0)


def balustrade(mb, x0, x1, y, z, mats, h=1.05, out=-1):
    """Cement vase-balustrade (white): base rail, square balusters at 0.24 m, top rail."""
    mb.box(x0, x1, y - 0.1, y + 0.1, z, z + 0.15, mats["trim"])
    mb.box(x0 - 0.04, x1 + 0.04, y - 0.13, y + 0.13, z + h - 0.15, z + h, mats["trim"])
    n = max(2, int((x1 - x0) / 0.24))
    for k in range(n + 1):
        xx = x0 + (x1 - x0) * k / n
        mb.box(xx - 0.055, xx + 0.055, y - 0.055, y + 0.055, z + 0.15, z + h - 0.15, mats["trim"])
    for xx in (x0, x1):
        mb.box(xx - 0.12, xx + 0.12, y - 0.12, y + 0.12, z, z + h + 0.05, mats["trim"])


def flat_roof(mb, W_, D, z, mats, parapet_mat, parapet_h=0.95, rebar=False):
    mb.box(-W_ / 2 - 0.08, W_ / 2 + 0.08, -D / 2 - 0.08, D / 2 + 0.08, z, z + 0.18, mats["roofdeck"])
    if parapet_h > 0:
        t = 0.14; zt = z + 0.18
        mb.box(-W_ / 2, W_ / 2, -D / 2, -D / 2 + t, zt, zt + parapet_h, parapet_mat)
        mb.box(-W_ / 2, W_ / 2, D / 2 - t, D / 2, zt, zt + parapet_h, parapet_mat)
        mb.box(-W_ / 2, -W_ / 2 + t, -D / 2 + t, D / 2 - t, zt, zt + parapet_h, parapet_mat)
        mb.box(W_ / 2 - t, W_ / 2, -D / 2 + t, D / 2 - t, zt, zt + parapet_h, parapet_mat)
        mb.box(-W_ / 2 - 0.05, W_ / 2 + 0.05, -D / 2 - 0.05, -D / 2 + t + 0.05, zt + parapet_h, zt + parapet_h + 0.06, mats["trim"])
    if rebar:                                           # rebar stubs on the column heads (house will grow a storey)
        for cx in np.linspace(-W_ / 2 + 0.2, W_ / 2 - 0.2, 3):
            for cy in np.linspace(-D / 2 + 0.2, D / 2 - 0.2, 3):
                for dx, dy in ((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)):
                    mb.cyl((cx + dx, cy + dy, z + 0.18), (cx + dx * 1.3, cy + dy * 1.3, z + 0.95), 0.011, mats["rust"], segs=5)


def stair_house(mb, x, y, z, mats, wall_mat, w=3.4, d=4.6, h=2.9, door_face=-1):
    hw, hd = w / 2, d / 2
    walls = [Wall((x - hw, y - hd, 0), (1, 0, 0), (0, -1, 0), w), Wall((x + hw, y + hd, 0), (-1, 0, 0), (0, 1, 0), w),
             Wall((x - hw, y + hd - T_WALL, 0), (0, -1, 0), (-1, 0, 0), d - 2 * T_WALL),
             Wall((x + hw, y - hd + T_WALL, 0), (0, 1, 0), (1, 0, 0), d - 2 * T_WALL)]
    for k, wl in enumerate(walls):
        ops = []
        if k == 0:
            ops = [(w / 2 - 0.5, w / 2 + 0.5, z, z + 2.1)]
        if k == 2:
            ops = [(0.9, 1.6, z + 1.2, z + 2.0)]
        wl.build(mb, z, z + h, ops, wall_mat)
        for (u0, u1, z0, z1) in ops:
            window_fill(mb, wl, u0, u1, z0, z1, mats, "door" if k == 0 else "win")
    mb.box(x - hw - 0.25, x + hw + 0.25, y - hd - 0.35, y + hd + 0.25, z + h, z + h + 0.16, mats["roofdeck"])
    return z + h + 0.16


def plinth_and_yard(mb, W_, D, mats, yard=6.0, yard_w=None):
    """Foundation down into the ground (houses never float on uneven ground) and a concrete drying yard."""
    mb.box(-W_ / 2 - 0.05, W_ / 2 + 0.05, -D / 2 - 0.05, D / 2 + 0.05, -2.5, 0.30, mats["plinth"])
    if yard > 0:
        yw = (yard_w or W_ + 2.0) / 2
        mb.box(-yw, yw, -D / 2 - yard, -D / 2 - 0.05, -1.2, 0.06, mats["pad"], skip=("bottom",))


def courtyard_wall(mb, W_, D, depth, mats, wall_mat, h=1.9, cap=None, gate=2.6):
    t = 0.24; y0 = -D / 2 - depth; xL, xR = -W_ / 2 - 0.6, W_ / 2 + 0.6
    mb.box(xL, xL + t, y0, -D / 2, -0.4, h, wall_mat)
    mb.box(xR - t, xR, y0, -D / 2, -0.4, h, wall_mat)
    g0, g1 = -gate / 2, gate / 2
    mb.box(xL, g0 - 0.3, y0, y0 + t, -0.4, h, wall_mat)
    mb.box(g1 + 0.3, xR, y0, y0 + t, -0.4, h, wall_mat)
    for gx in (g0 - 0.3, g1):                                       # gate piers
        mb.box(gx, gx + 0.3, y0 - 0.05, y0 + t + 0.05, -0.4, h + 0.45, wall_mat)
    mb.box(g0, g1, y0 + 0.08, y0 + 0.14, 0.0, h + 0.1, mats["door"])
    if cap is not None:
        for (a, b, c0, c1) in ((xL, xL + t, y0, -D / 2), (xR - t, xR, y0, -D / 2), (xL, g0 - 0.3, y0, y0 + t), (g1 + 0.3, xR, y0, y0 + t)):
            mb.box(a - 0.06, b + 0.06, c0 - 0.06, c1 + 0.06, h, h + 0.1, cap)


# ------------------------------------------------------------------ the five house types
def house_modern(mats, rng, floors=3, W_=11.0, D=12.0, narrow=False, hat_roof=False):
    """Concrete-frame house: tiled front facade with balconies, bare red brick sides and back, flat roof."""
    mb = MB()
    plinth_and_yard(mb, W_, D, mats, yard=rng.uniform(5, 8))
    H = H_STOREY; z_top = floors * H
    front_mat = mats["facade"]; side_mat = mats["brick"] if rng.random() < 0.7 else mats["facade"]
    n_bays = 2 if narrow else 3
    front = Wall((-W_ / 2, -D / 2, 0), (1, 0, 0), (0, -1, 0), W_)
    back = Wall((W_ / 2, D / 2, 0), (-1, 0, 0), (0, 1, 0), W_)
    left = Wall((-W_ / 2, D / 2 - T_WALL, 0), (0, -1, 0), (-1, 0, 0), D - 2 * T_WALL)
    right = Wall((W_ / 2, -D / 2 + T_WALL, 0), (0, 1, 0), (1, 0, 0), D - 2 * T_WALL)
    fops = []; fills = []
    for f in range(floors):
        z0 = f * H
        if f == 0:
            bw = bays(W_, n_bays, 2.6 if not narrow else 2.4)
            for k, (u0, u1) in enumerate(bw):
                if k == n_bays // 2 or narrow and k == 0:
                    fops.append((u0, u1, 0.3, 3.0)); fills.append(("shutter" if rng.random() < 0.6 else "door", u0, u1, 0.3, 3.0, False))
                else:
                    fops.append((u0 + 0.5, u1 - 0.5, 1.2, 2.6)); fills.append(("win", u0 + 0.5, u1 - 0.5, 1.2, 2.6, True))
        else:
            for k, (u0, u1) in enumerate(bays(W_, n_bays, 2.4 if not narrow else 2.2)):
                door = k == n_bays // 2 or (narrow and k == 0)
                oz0 = z0 + (0.0 if door else 0.9) + 0.15
                fops.append((u0, u1, oz0, z0 + 2.55)); fills.append(("win", u0, u1, oz0, z0 + 2.55, False))
    front.build(mb, 0.3, z_top, fops, front_mat)
    for kind, u0, u1, z0, z1, g in fills:
        window_fill(mb, front, u0, u1, z0, z1, mats, kind, grille=g)
    for wl, mat, nb in ((back, side_mat, n_bays), (left, side_mat, 2), (right, side_mat, 2)):
        ops = []
        for f in range(floors):
            z0 = f * H
            for (u0, u1) in bays(wl.L, nb, 1.2):
                if rng.random() < 0.8:
                    ops.append((u0, u1, z0 + 1.1, z0 + 2.4))
        wl.build(mb, 0.3, z_top, ops, mat)
        for (u0, u1, z0, z1) in ops:
            window_fill(mb, wl, u0, u1, z0, z1, mats, "win", grille=(z0 < 2 and rng.random() < 0.6))
        if rng.random() < 0.5 and ops:
            u0, u1, z0, z1 = ops[int(rng.integers(len(ops)))]
            ac_unit(mb, wl, u1 + 0.6, z0 + 0.1, mats)
    # floor bands (slab edges) and balconies on the front
    for f in range(1, floors + 1):
        zf = f * H
        mb.box(-W_ / 2 - 0.06, W_ / 2 + 0.06, -D / 2 - 0.06, D / 2 + 0.06, zf - 0.16, zf, mats["trim"])
    bal_full = rng.random() < 0.5
    for f in range(1, floors):
        zf = f * H; dep = 1.3
        if bal_full:
            bx0, bx1 = -W_ / 2, W_ / 2
        else:
            (u0, u1) = bays(W_, n_bays, 2.4 if not narrow else 2.2)[n_bays // 2 if not narrow else 0]
            bx0, bx1 = -W_ / 2 + u0 - 0.8, -W_ / 2 + u1 + 0.8
        mb.box(bx0, bx1, -D / 2 - dep, -D / 2, zf - 0.16, zf, mats["trim"])
        if rng.random() < 0.55:
            balustrade(mb, bx0 + 0.12, bx1 - 0.12, -D / 2 - dep + 0.12, zf, mats)
            for xx in (bx0 + 0.12, bx1 - 0.12):              # side returns: base and hand rail back to the wall
                mb.box(xx - 0.06, xx + 0.06, -D / 2 - dep + 0.12, -D / 2, zf, zf + 0.15, mats["trim"])
                mb.box(xx - 0.07, xx + 0.07, -D / 2 - dep + 0.12, -D / 2, zf + 0.9, zf + 1.05, mats["trim"])
                mb.box(xx - 0.05, xx + 0.05, -D / 2 - 0.7, -D / 2 - 0.6, zf + 0.15, zf + 0.9, mats["trim"])
        else:
            mb.box(bx0, bx1, -D / 2 - dep, -D / 2 - dep + 0.12, zf, zf + 1.05, front_mat)
            mb.box(bx0, bx0 + 0.12, -D / 2 - dep, -D / 2, zf, zf + 1.05, front_mat)
            mb.box(bx1 - 0.12, bx1, -D / 2 - dep, -D / 2, zf, zf + 1.05, front_mat)
            mb.box(bx0 - 0.03, bx1 + 0.03, -D / 2 - dep - 0.03, -D / 2 - dep + 0.15, zf + 1.05, zf + 1.12, mats["trim"])
    # roof
    if hat_roof:
        zr = gable_roof(mb, -W_ / 2, W_ / 2, -D / 2, D / 2, z_top + 0.4, 28, 0.6, 0.4, dict(mats, roof=mats["roof_red"], ridge=mats["roof_red"]), front_mat)
        mb.box(-W_ / 2, W_ / 2, -D / 2, D / 2, z_top, z_top + 0.4, front_mat)
    else:
        flat_roof(mb, W_, D, z_top, mats, front_mat)
        zs = stair_house(mb, W_ / 2 - 1.9, D / 2 - 2.5, z_top + 0.18, mats, side_mat)
        if rng.random() < 0.85:
            solar_heater(mb, -W_ / 2 + 2.6, D / 2 - 1.2, z_top + 0.18, mats, length=float(rng.choice([1.5, 1.8, 2.1])))
        if rng.random() < 0.7:
            water_tank(mb, W_ / 2 - 1.9, D / 2 - 2.5, zs + 0.25, mats)
    return mb


def house_unfinished(mats, rng, floors=3, W_=10.5, D=12.0):
    """Exposed concrete frame with red-brick infill, unglazed openings on the top floor, rebar stubs."""
    mb = MB()
    plinth_and_yard(mb, W_, D, mats, yard=rng.uniform(3, 6))
    H = H_STOREY; z_top = floors * H
    walls = [Wall((-W_ / 2, -D / 2, 0), (1, 0, 0), (0, -1, 0), W_), Wall((W_ / 2, D / 2, 0), (-1, 0, 0), (0, 1, 0), W_),
             Wall((-W_ / 2, D / 2 - T_WALL, 0), (0, -1, 0), (-1, 0, 0), D - 2 * T_WALL),
             Wall((W_ / 2, -D / 2 + T_WALL, 0), (0, 1, 0), (1, 0, 0), D - 2 * T_WALL)]
    for k, wl in enumerate(walls):
        ops = []; fills = []
        nb = 3 if k < 2 else 2
        for f in range(floors):
            z0 = f * H
            for (u0, u1) in bays(wl.L, nb, 1.5 if k < 2 else 1.1):
                if k == 0 and f == 0 and abs((u0 + u1) / 2 - wl.L / 2) < 1:
                    ops.append((u0 - 0.4, u1 + 0.4, 0.3, 2.9)); fills.append("shutter")
                elif rng.random() < 0.85:
                    ops.append((u0, u1, z0 + 1.0, z0 + 2.5))
                    fills.append("hole" if (f == floors - 1 and rng.random() < 0.7) else "win")
        wl.build(mb, 0.3, z_top, ops, mats["brick"])
        for (u0, u1, z0, z1), kind in zip(ops, fills):
            window_fill(mb, wl, u0, u1, z0, z1, mats, kind, grille=(kind == "win" and z0 < 2))
        # exposed columns at the corners and mid-span, proud of the brick
        for uc in (0.0, wl.L / 2, wl.L):
            wl.block(mb, uc - 0.2, uc + 0.2, 0.3, z_top, mats["concrete"], d0=-0.04, d1=0.1)
    for f in range(1, floors + 1):
        zf = f * H
        mb.box(-W_ / 2 - 0.05, W_ / 2 + 0.05, -D / 2 - 0.05, D / 2 + 0.05, zf - 0.35, zf, mats["concrete"])
    flat_roof(mb, W_, D, z_top, mats, mats["brick"], parapet_h=0.0 if rng.random() < 0.5 else 0.6, rebar=True)
    if rng.random() < 0.6:
        water_tank(mb, -W_ / 2 + 1.5, D / 2 - 1.5, z_top + 0.43, mats)
    if rng.random() < 0.5:
        solar_heater(mb, W_ / 2 - 2.2, D / 2 - 1.2, z_top + 0.18, mats)
    return mb


def house_rendered(mats, rng, W_=10.0, D=8.5):
    """Two-storey rendered brick house with a grey pantile gable roof, small windows and a lean-to shed."""
    mb = MB()
    plinth_and_yard(mb, W_, D, mats, yard=rng.uniform(4, 7))
    H = 3.0; floors = 2; z_top = floors * H
    walls = [Wall((-W_ / 2, -D / 2, 0), (1, 0, 0), (0, -1, 0), W_), Wall((W_ / 2, D / 2, 0), (-1, 0, 0), (0, 1, 0), W_),
             Wall((-W_ / 2, D / 2 - T_WALL, 0), (0, -1, 0), (-1, 0, 0), D - 2 * T_WALL),
             Wall((W_ / 2, -D / 2 + T_WALL, 0), (0, 1, 0), (1, 0, 0), D - 2 * T_WALL)]
    for k, wl in enumerate(walls):
        ops = []; fills = []
        nb = 3 if k < 2 else 1
        for f in range(floors):
            for j, (u0, u1) in enumerate(bays(wl.L, nb, 1.0)):
                if k == 0 and f == 0 and j == 1:
                    ops.append((u0 - 0.2, u1 + 0.2, 0.3, 2.4)); fills.append("door")
                elif k < 2 or rng.random() < 0.5:
                    ops.append((u0, u1, f * H + 1.0, f * H + 2.2)); fills.append("win")
        wl.build(mb, 0.3, z_top, ops, mats["plaster"])
        for (u0, u1, z0, z1), kind in zip(ops, fills):
            window_fill(mb, wl, u0, u1, z0, z1, mats, kind, grille=(kind == "win" and z0 < 2 and rng.random() < 0.7))
    mb.box(-W_ / 2 - 0.04, W_ / 2 + 0.04, -D / 2 - 0.04, D / 2 + 0.04, H - 0.12, H, mats["trim"])
    gable_roof(mb, -W_ / 2, W_ / 2, -D / 2, D / 2, z_top, 27, 0.55, 0.35, mats, mats["plaster"])
    # lean-to shed on one gable end
    sx = W_ / 2; sw = 3.2; sd = D * 0.8
    mb.box(sx, sx + sw, -sd / 2, sd / 2, -0.3, 2.4, mats["brick"] if rng.random() < 0.6 else mats["plaster"])
    tn = math.tan(math.radians(14))
    c = [(sx, -sd / 2 - 0.3, 2.9), (sx + sw + 0.35, -sd / 2 - 0.3, 2.9 - (sw + 0.35) * tn),
         (sx + sw + 0.35, sd / 2 + 0.3, 2.9 - (sw + 0.35) * tn), (sx, sd / 2 + 0.3, 2.9)]
    mb.hexa(c + [(x, y, z + 0.1) for (x, y, z) in c], mats["roof"])
    mb.box(sx, sx + sw, -sd / 2, sd / 2, 2.4, 2.9 - sw * tn * 0.5, mats["plaster"], skip=("top",))
    if rng.random() < 0.5:
        courtyard_wall(mb, W_, D, rng.uniform(5, 7), mats, mats["plaster"], cap=mats["roof"])
    return mb


def house_old(mats, rng, W_=11.0, D=7.0):
    """Single-storey grey-brick farmhouse (qing zhuan), deep eaves, grey pantile roof, walled yard."""
    mb = MB()
    plinth_and_yard(mb, W_, D, mats, yard=0.0)
    H = 3.4
    walls = [Wall((-W_ / 2, -D / 2, 0), (1, 0, 0), (0, -1, 0), W_), Wall((W_ / 2, D / 2, 0), (-1, 0, 0), (0, 1, 0), W_),
             Wall((-W_ / 2, D / 2 - T_WALL, 0), (0, -1, 0), (-1, 0, 0), D - 2 * T_WALL),
             Wall((W_ / 2, -D / 2 + T_WALL, 0), (0, 1, 0), (1, 0, 0), D - 2 * T_WALL)]
    for k, wl in enumerate(walls):
        ops = []; fills = []
        if k == 0:
            for j, (u0, u1) in enumerate(bays(W_, 3, 0.9)):
                if j == 1:
                    ops.append((u0 - 0.1, u1 + 0.1, 0.3, 2.5)); fills.append("door")
                else:
                    ops.append((u0, u1, 1.3, 2.3)); fills.append("win")
        elif k == 1 and rng.random() < 0.6:
            ops.append((W_ / 2 - 0.4, W_ / 2 + 0.4, 1.6, 2.4)); fills.append("win")
        wl.build(mb, 0.3, H, ops, mats["brick_grey"])
        for (u0, u1, z0, z1), kind in zip(ops, fills):
            window_fill(mb, wl, u0, u1, z0, z1, mats, kind, grille=(kind == "win"))
    gable_roof(mb, -W_ / 2, W_ / 2, -D / 2, D / 2, H, 30, 0.75, 0.3, mats, mats["brick_grey"])
    courtyard_wall(mb, W_, D, rng.uniform(6, 9), mats, mats["brick_grey"], h=2.0, cap=mats["roof"])
    mb.box(-W_ / 2 - 0.5, W_ / 2 + 0.5, -D / 2 - 6.5, -D / 2 - 0.05, -1.0, 0.05, mats["pad"], skip=("bottom",))
    return mb


TYPES = [  # (name, builder, weight in a village, weight in a hamlet)
    ("A_tile3", lambda m, r: house_modern(m, r, floors=3, W_=r.uniform(10, 12.5), D=r.uniform(11, 13)), 0.30, 0.15),
    ("B_tile4", lambda m, r: house_modern(m, r, floors=4, W_=r.uniform(6.2, 7.2), D=r.uniform(12, 14), narrow=True,
                                          hat_roof=r.random() < 0.5), 0.17, 0.05),
    ("C_frame3", lambda m, r: house_unfinished(m, r, floors=int(r.choice([2, 3, 3])), W_=r.uniform(9.5, 11.5)), 0.18, 0.15),
    ("D_render2", lambda m, r: house_rendered(m, r, W_=r.uniform(9, 11.5), D=r.uniform(8, 9.5)), 0.20, 0.35),
    ("E_old1", lambda m, r: house_old(m, r, W_=r.uniform(10, 13), D=r.uniform(6.5, 7.5)), 0.15, 0.30),
]
N_VARIANTS = 3          # seeded variants per type -> 15 distinct house meshes


# ------------------------------------------------------------------ materials
def _tex_nodes(nt, path, size_m, noncolor=False, blend=0.2):
    L = nt.links.new
    tc = nt.nodes.get("vil_tc") or nt.nodes.new("ShaderNodeTexCoord")
    tc.name = "vil_tc"
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / size_m,) * 3
    L(tc.outputs["Object"], mp.inputs["Vector"])
    n = nt.nodes.new("ShaderNodeTexImage")
    im = bpy.data.images.load(path, check_existing=True)
    if noncolor:
        im.colorspace_settings.name = "Non-Color"
    n.image = im; n.projection = "BOX"; n.projection_blend = blend; n.interpolation = "Cubic"
    L(mp.outputs[0], n.inputs[0])
    return n


def _grime(nt, col_socket, strength=0.28, splash=0.25):
    """Rain streaks (vertically stretched noise, darker under ledges) and ground splash on a colour socket."""
    L = nt.links.new
    tc = nt.nodes.get("vil_tc")
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (2.2, 2.2, 0.16)
    L(tc.outputs["Object"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.0; nz.inputs["Detail"].default_value = 4.0
    L(mp.outputs[0], nz.inputs["Vector"])
    st = nt.nodes.new("ShaderNodeMapRange"); st.inputs[1].default_value = 0.5; st.inputs[2].default_value = 0.78
    L(nz.outputs["Fac"], st.inputs[0])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sep.inputs[0])
    sp = nt.nodes.new("ShaderNodeMapRange"); sp.inputs[1].default_value = 0.2; sp.inputs[2].default_value = 1.1
    sp.inputs[3].default_value = 1.0; sp.inputs[4].default_value = 0.0
    L(sep.outputs["Z"], sp.inputs[0])
    a = nt.nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY"; a.inputs[1].default_value = strength; L(st.outputs[0], a.inputs[0])
    b = nt.nodes.new("ShaderNodeMath"); b.operation = "MULTIPLY_ADD"; b.inputs[1].default_value = splash
    L(sp.outputs[0], b.inputs[0]); L(a.outputs[0], b.inputs[2])
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0; inv.use_clamp = True
    L(b.outputs[0], inv.inputs[1])
    mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    L(col_socket, mul.inputs["A"])
    cmb = nt.nodes.new("ShaderNodeCombineColor"); L(inv.outputs[0], cmb.inputs[0]); L(inv.outputs[0], cmb.inputs[1]); L(inv.outputs[0], cmb.inputs[2])
    L(cmb.outputs[0], mul.inputs["B"])
    return mul.outputs["Result"]


def _instance_var(nt, col_socket, lo=0.85, hi=1.12, palette=None):
    """Per-instance value variation (and an optional colour palette picked by Object Info > Random)."""
    L = nt.links.new
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    if palette:
        cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.interpolation = "CONSTANT"
        els = cr.color_ramp.elements
        while len(els) < len(palette):
            els.new(0.5)
        for i, c in enumerate(palette):
            els[i].position = i / len(palette); els[i].color = (*c, 1)
        L(oi.outputs["Random"], cr.inputs[0])
        tint = nt.nodes.new("ShaderNodeMix"); tint.data_type = "RGBA"; tint.blend_type = "MULTIPLY"; tint.inputs["Factor"].default_value = 1.0
        L(col_socket, tint.inputs["A"]); L(cr.outputs[0], tint.inputs["B"])
        col_socket = tint.outputs["Result"]
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs[3].default_value = lo; mr.inputs[4].default_value = hi
    rnd = nt.nodes.new("ShaderNodeMath"); rnd.operation = "FRACT"
    mm = nt.nodes.new("ShaderNodeMath"); mm.operation = "MULTIPLY"; mm.inputs[1].default_value = 7.31
    L(oi.outputs["Random"], mm.inputs[0]); L(mm.outputs[0], rnd.inputs[0]); L(rnd.outputs[0], mr.inputs[0])
    sc = nt.nodes.new("ShaderNodeVectorMath"); sc.operation = "SCALE"
    L(col_socket, sc.inputs[0]); L(mr.outputs[0], sc.inputs["Scale"])
    return sc.outputs[0]


def _textured(name, tex, size_m, tint=None, normalise=False, rough_scale=1.0, rough_add=0.0, bump=0.4, grime=True,
              palette=None, var=(0.85, 1.12), coat=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial"); p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(p.outputs[0], out.inputs[0])
    d = _tex_nodes(nt, tex["diff"], size_m)
    col = d.outputs["Color"]
    if normalise:
        # use the photo only for its pattern: luminance / its mean, multiplied by the wanted colour
        bw = nt.nodes.new("ShaderNodeRGBToBW"); L(col, bw.inputs[0])
        dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[1].default_value = normalise
        L(bw.outputs[0], dv.inputs[0])
        cm = nt.nodes.new("ShaderNodeCombineColor"); L(dv.outputs[0], cm.inputs[0]); L(dv.outputs[0], cm.inputs[1]); L(dv.outputs[0], cm.inputs[2])
        col = cm.outputs[0]
    if tint:
        t = nt.nodes.new("ShaderNodeMix"); t.data_type = "RGBA"; t.blend_type = "MULTIPLY"; t.inputs["Factor"].default_value = 1.0
        L(col, t.inputs["A"]); t.inputs["B"].default_value = (*tint, 1); col = t.outputs["Result"]
    col = _instance_var(nt, col, *var, palette=palette)
    if grime:
        col = _grime(nt, col)
    L(col, p.inputs["Base Color"])
    r = _tex_nodes(nt, tex["rough"], size_m, noncolor=True)
    rm = nt.nodes.new("ShaderNodeMath"); rm.operation = "MULTIPLY_ADD"; rm.inputs[1].default_value = rough_scale
    rm.inputs[2].default_value = rough_add; rm.use_clamp = True
    L(r.outputs["Color"], rm.inputs[0]); L(rm.outputs[0], p.inputs["Roughness"])
    bw2 = nt.nodes.new("ShaderNodeRGBToBW"); L(d.outputs["Color"], bw2.inputs[0])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump; bp.inputs["Distance"].default_value = 0.02
    L(bw2.outputs[0], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    if coat:
        p.inputs["Coat Weight"].default_value = coat; p.inputs["Coat Roughness"].default_value = 0.08
    return m


def _plain(name, col, metal=0.0, rough=0.5, coat=0.0):
    from . import materials as M
    return M.principled(name, col, metal, rough, coat)


def _glass():
    m = bpy.data.materials.get("VilGlass")
    if m:
        return m
    m = bpy.data.materials.new("VilGlass"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial"); p = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(p.outputs[0], out.inputs[0])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # per-window variation: dark room / blue-green reflective film / pale curtains (cells ~1.5 m)
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 0.7
    L(tc.outputs["Object"], vor.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.interpolation = "CONSTANT"
    e = cr.color_ramp.elements
    e[0].position = 0.0; e[0].color = (0.012, 0.014, 0.016, 1)
    e[1].position = 0.45; e[1].color = (0.020, 0.060, 0.058, 1)
    e3 = e.new(0.75); e3.color = (0.30, 0.27, 0.22, 1)
    L(vor.outputs["Color"], cr.inputs[0])
    L(cr.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.6
    p.inputs["Coat Weight"].default_value = 1.0; p.inputs["Coat Roughness"].default_value = 0.02; p.inputs["Coat IOR"].default_value = 1.52
    return m


def _shutter():
    m = bpy.data.materials.get("VilShutter")
    if m:
        return m
    m = bpy.data.materials.new("VilShutter"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial"); p = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(p.outputs[0], out.inputs[0])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "Z"
    wv.inputs["Scale"].default_value = 3.7     # period 2pi/(20 s) = 0.085 m; wv.inputs["Distortion"].default_value = 0.0
    L(tc.outputs["Object"], wv.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.6; bp.inputs["Distance"].default_value = 0.01
    L(wv.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    p.inputs["Base Color"].default_value = (0.42, 0.44, 0.45, 1); p.inputs["Metallic"].default_value = 0.7
    p.inputs["Roughness"].default_value = 0.45
    return m


def materials(assets):
    t = os.path.join(assets, "tex")
    def tp(aid):
        return dict(diff=f"{t}/{aid}/{aid}_diff_2k.jpg", rough=f"{t}/{aid}/{aid}_rough_2k.jpg")
    facade_palette = [(1.0, 1.0, 1.0), (1.0, 0.93, 0.80), (0.98, 0.86, 0.82), (0.86, 0.91, 0.96), (1.0, 0.97, 0.92)]
    return dict(
        facade=_textured("VilFacadeTile", tp("rectangular_facade_tiles"), 2.0, tint=(0.72, 0.71, 0.68), normalise=0.093,
                         rough_scale=0.5, rough_add=0.05, bump=0.25, palette=facade_palette, coat=0.3),
        brick=_textured("VilBrick", tp("red_bricks_04"), 2.5, tint=(1.6, 1.6, 1.6), bump=0.5),
        brick_grey=_textured("VilBrickGrey", tp("red_bricks_04"), 2.5, tint=(0.24, 0.25, 0.26), normalise=0.063, bump=0.5,
                             var=(0.8, 1.15)),
        plaster=_textured("VilPlaster", tp("plaster_grey_04"), 1.5, tint=(1.0, 0.98, 0.95), bump=0.3,
                          palette=[(1.0, 1.0, 1.0), (0.92, 0.90, 0.86), (1.05, 1.0, 0.92)], var=(0.8, 1.15)),
        concrete=_textured("VilConcrete", tp("concrete_wall_008"), 2.7, bump=0.3),
        plinth=_textured("VilPlinth", tp("concrete_wall_008"), 2.7, tint=(0.75, 0.74, 0.72), bump=0.3),
        pad=_textured("VilPad", tp("concrete_wall_008"), 2.7, tint=(0.95, 0.93, 0.9), bump=0.2, grime=False),
        roofdeck=_textured("VilRoofDeck", tp("concrete_wall_008"), 2.7, tint=(0.55, 0.54, 0.52), bump=0.3),
        roof=_textured("VilRoofGrey", tp("grey_roof_tiles"), 3.0, bump=0.8, grime=False, var=(0.8, 1.1)),
        ridge=_textured("VilRidge", tp("grey_roof_tiles"), 3.0, tint=(0.7, 0.7, 0.7), bump=0.5, grime=False),
        roof_red=_textured("VilRoofRed", tp("clay_roof_tiles_02"), 2.5, tint=(0.85, 0.80, 0.78), bump=0.8, grime=False, coat=0.4),
        trim=_textured("VilTrim", tp("plaster_grey_04"), 1.5, tint=(1.9, 1.88, 1.82), bump=0.2, var=(0.9, 1.05)),
        glass=_glass(), shutter=_shutter(),
        frame=_plain("VilFrame", (0.70, 0.71, 0.72), 1.0, 0.35),
        sill=_plain("VilSill", (0.55, 0.54, 0.52), 0.0, 0.6),
        steel=_plain("VilSteel", (0.62, 0.62, 0.63), 1.0, 0.28),
        rust=_plain("VilRebar", (0.16, 0.07, 0.035), 0.3, 0.8),
        door=_plain("VilDoor", (0.20, 0.03, 0.022), 0.4, 0.45, coat=0.4),
        interior=_plain("VilInterior", (0.012, 0.011, 0.010), 0.0, 0.95),
        tank_black=_plain("VilTankBlack", (0.018, 0.018, 0.02), 0.0, 0.45),
        tank_steel=_plain("VilTankSteel", (0.74, 0.75, 0.76), 1.0, 0.22),
        tube=_plain("VilSolarTube", (0.010, 0.012, 0.016), 0.0, 0.05, coat=1.0),
        ac=_plain("VilAC", (0.80, 0.80, 0.78), 0.0, 0.5),
    )


# ------------------------------------------------------------------ templates and placement
def templates(assets):
    """Build the 15 house meshes once (5 types x 3 seeded variants)."""
    out = {}
    mats = materials(assets)
    order = list(dict.fromkeys(mats.values()))
    idx = {mt.name: i for i, mt in enumerate(order)}
    for ti, (tname, fn, _, _) in enumerate(TYPES):
        for v in range(N_VARIANTS):
            name = f"House_{tname}_{v}"
            me = bpy.data.meshes.get(name)
            if me is None:
                rng = np.random.default_rng(1000 + 17 * ti + v)
                mb = fn(mats, rng)
                mb.m = [idx[mt.name] for mt in mb.m]
                me = mb.make(name, order)
            out.setdefault(tname, []).append(me)
    return out


SITES = []              # every placed house (x, y, ground_z, yaw): vegetation keeps clear of these


def place(coll, assets, sites, rng, hamlet=False, name="Village"):
    """Instance houses at sites [(x, y, ground_z, yaw), ...] (linked mesh data: one copy in memory)."""
    tpl = templates(assets)
    SITES.extend(sites)
    w = np.array([t[3] if hamlet else t[2] for t in TYPES]); w = w / w.sum()
    obs = []
    for k, (x, y, g, yaw) in enumerate(sites):
        ti = int(rng.choice(len(TYPES), p=w))
        me = tpl[TYPES[ti][0]][int(rng.integers(N_VARIANTS))]
        ob = bpy.data.objects.new(f"{name}_{k:02d}", me)
        ob.location = (x, y, g); ob.rotation_euler = (0, 0, yaw)
        coll.objects.link(ob); obs.append(ob)
    return obs
