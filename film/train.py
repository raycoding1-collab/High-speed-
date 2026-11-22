"""CR450AF trainset geometry (bpy).

Object conventions: every car object has its origin midway between its two
bogie pivots, at top-of-rail height; local +x points to the train's front,
+y to the left, +z up.  Design data uses X = distance from the nose tip
(for end cars) or from the car's front end (middle cars).
"""
import math

import bpy
import numpy as np

from . import specs as S
from . import trainshape as TS
from .bl import new_mesh_object, grid_faces

GAP = 0.30           # half of the body-to-body gap closed by the gangway fairing


def car_layout():
    """Return list of dicts describing each car (index 1..8)."""
    cars = []
    x_front = 0.0            # coupling-plane position along the train, measured rearwards from the nose tip
    for i in range(1, S.N_CARS + 1):
        L = S.END_CAR_LENGTH if i in (1, S.N_CARS) else S.MID_CAR_LENGTH
        if i == 1:
            b1 = S.END_FRONT_BOGIE_X; b2 = b1 + S.TRUCK_CENTRES
            kind = "end"
        elif i == S.N_CARS:
            b2 = L - S.END_FRONT_BOGIE_X; b1 = b2 - S.TRUCK_CENTRES
            kind = "end_rev"
        else:
            b1 = (L - S.TRUCK_CENTRES) / 2; b2 = b1 + S.TRUCK_CENTRES
            kind = "mid"
        cars.append(dict(index=i, kind=kind, length=L, start=x_front,
                         bogies=(x_front + b1, x_front + b2), centre=x_front + (b1 + b2) / 2,
                         motor=S.FORMATION[i - 1] == "M", panto=i in S.PANTO_CARS))
        x_front += L
    return cars


# ------------------------------------------------------------------ shell meshes
def _vrow(ring_len, nsec):
    out = []
    for j in range(ring_len):
        out.append(0.5 * j / (nsec - 1) if j < nsec else 1.0 - 0.5 * (ring_len - j) / (nsec - 1))
    return out


def _shell(name, X_st, sections, X_origin, L_uv, tip=False, cap_front=False, sharp_rows=None, coll=None):
    """Closed tube through rings at stations X (from front); local x = X_origin - X.

    UV: u = X / L_uv, v = 0..0.5 right half (bottom->top), 0.5..1 left half (top->bottom).
    """
    nsec = TS.NSEC
    ring_len = 2 * nsec - 2
    vrow = _vrow(ring_len, nsec)
    rings = [TS.full_ring(hs) for hs in sections]
    verts = []
    idx = []           # idx[r][j] -> vertex index
    for r, (X, ring) in enumerate(zip(X_st, rings)):
        if tip and r == 0:
            verts.append((X_origin - X, 0.0, float(ring[0, 1])))
            idx.append([0] * ring_len)
            continue
        base = len(verts)
        verts.extend((X_origin - X, float(p[0]), float(p[1])) for p in ring)
        idx.append(list(range(base, base + ring_len)))
    faces, fuv = [], []
    for r in range(len(X_st) - 1):
        u0 = X_st[r] / L_uv; u1 = X_st[r + 1] / L_uv
        for j in range(ring_len):
            j2 = (j + 1) % ring_len
            v0 = vrow[j]; v1 = vrow[j2] if j2 != 0 else 1.0
            a, b, c, d = idx[r][j], idx[r][j2], idx[r + 1][j2], idx[r + 1][j]
            if tip and r == 0:
                faces.append((a, c, d)); fuv.append(((u0, 0.5 * (v0 + v1)), (u1, v1), (u1, v0)))
            else:
                faces.append((a, b, c, d)); fuv.append(((u0, v0), (u0, v1), (u1, v1), (u1, v0)))
    # rear cap (and optional front cap): fan to centre
    def cap(r, flip):
        hs = sections[r]
        c = len(verts)
        verts.append((X_origin - X_st[r], 0.0, 0.5 * (float(hs[0, 1]) + float(hs[-1, 1]))))
        for j in range(ring_len):
            j2 = (j + 1) % ring_len
            f = (idx[r][j2], idx[r][j], c) if not flip else (idx[r][j], idx[r][j2], c)
            faces.append(f); fuv.append(((0.999, 0.999),) * 3)
    cap(len(X_st) - 1, False)
    if cap_front:
        cap(0, True)
    sharp = []
    if sharp_rows is not None:
        for jj in (TS.CREASE_IDX, ring_len - TS.CREASE_IDX):
            for r in range(len(X_st) - 1):
                if sharp_rows[r] and sharp_rows[r + 1]:
                    sharp.append((idx[r][jj], idx[r + 1][jj]))
    uvs = [uv for f in fuv for uv in f]
    return new_mesh_object(name, np.array(verts), faces, coll, uvs=np.array(uvs), sharp_edges=sharp, creases=sharp)


def build_end_car_shell(name="CarEnd_shell", coll=None):
    st = TS.stations_end_car()
    sections = [TS.half_section(x) for x in st]
    X_origin = S.END_FRONT_BOGIE_X + S.TRUCK_CENTRES / 2          # local x of the nose tip
    led = TS.rails(st)["ledge"]
    return _shell(name, st, sections, X_origin, S.END_CAR_LENGTH, tip=True, sharp_rows=led > 0.015, coll=coll)


def build_mid_car_shell(name="CarMid_shell", coll=None):
    L = S.MID_CAR_LENGTH
    st = np.unique(np.concatenate([np.arange(GAP, L - GAP + 1e-6, 0.5), [L - GAP]]))
    hs = TS.body_section()
    return _shell(name, st, [hs] * len(st), L / 2, L, cap_front=True, coll=coll)


# ------------------------------------------------------------------ full trainset
def bake_liveries(texdir):
    from . import livery
    import os
    out = {}
    for key, kind, kw in (("end", "end", {}), ("mid", "mid", dict(car_no=2)), ("mid4", "mid", dict(car_no=4)),
                          ("dining", "mid", dict(car_no=5, dining=True))):
        c = os.path.join(texdir, f"{key}_col.png")
        if not os.path.exists(c):
            livery.bake(kind, texdir, key, **kw)
        out[key] = (c, os.path.join(texdir, f"{key}_dat.png"))
    return out


def build_train(texdir, coll=None, raised_panto_car=6, with_interior=True):
    """Build the 8-car set at rest on straight track, nose of car 1 at +x.

    Returns dict(cars=[obj..], bogies=[(obj, car_idx, along)], wheelsets=[obj], pantos=[...]).
    'along' = distance of the pivot from the nose tip of car 1 (positive rearwards).
    """
    from . import materials as M
    from . import trainparts as P
    coll = coll or bpy.context.scene.collection
    tex = bake_liveries(texdir)
    mats = {k: M.body_material(f"Body_{k}", *v) for k, v in tex.items()}
    end_mesh = build_end_car_shell("CarEnd_mesh", coll)
    end_mesh.data.name = "CarEnd_mesh"
    mid_mesh = build_mid_car_shell("CarMid_mesh", coll)
    meshes = dict(end=end_mesh.data, mid=mid_mesh.data)
    bpy.data.objects.remove(end_mesh, do_unlink=True); bpy.data.objects.remove(mid_mesh, do_unlink=True)
    for me in meshes.values():
        me.materials.append(mats["end"])          # slot placeholder; real material linked per object
    seat_row = P.build_seat_row("SeatRow", coll); seat_row.hide_render = True; seat_row.hide_viewport = True
    ws_t = P.build_wheelset("Wheelset_T", coll, motor=False); ws_m = P.build_wheelset("Wheelset_M", coll, motor=True)
    bg_t = P.build_bogie("Bogie_T", coll, motor=False); bg_m = P.build_bogie("Bogie_M", coll, motor=True)
    for o in (ws_t, ws_m, bg_t, bg_m):
        o.hide_render = True; o.hide_viewport = True
    res = dict(cars=[], bogies=[], wheelsets=[], pantos=[], layout=car_layout())
    for c in res["layout"]:
        i = c["index"]
        kind = "end" if c["kind"] in ("end", "end_rev") else "mid"
        ob = bpy.data.objects.new(f"Car{i}", meshes[kind])
        coll.objects.link(ob)
        ob.material_slots[0].link = "OBJECT"
        key = "end" if kind == "end" else ("dining" if i == 5 else ("mid4" if i == 4 else "mid"))
        ob.material_slots[0].material = mats[key]
        ob.location = (-c["centre"], 0, 0)
        if c["kind"] == "end_rev":
            ob.rotation_euler = (0, 0, math.pi)
        # head/tail lamps: car 1 leads (white), car 8 trails (red)
        ob["lamp"] = (14.0, 13.5, 12.5) if i == 1 else ((9.0, 0.05, 0.03) if i == S.N_CARS else (0, 0, 0))
        mod = ob.modifiers.new("subd", "SUBSURF"); mod.levels = 0; mod.render_levels = 1 if kind == "end" else 0
        res["cars"].append(ob)
        L = c["length"]
        # interior
        if with_interior:
            if kind == "mid":
                rows = np.arange(-L / 2 + 4.6, L / 2 - 4.4, 1.02)
                xin = 0.0
            else:
                rows = np.arange(-L / 2 + 15.0, L / 2 - 4.0, 1.25) if True else []
                xin = 0.0
            if kind == "end":
                X_origin = S.END_FRONT_BOGIE_X + S.TRUCK_CENTRES / 2
                rows = np.arange(X_origin - 25.0, X_origin - 10.2, 1.25)
                ilen = 15.0; icx = X_origin - 17.6
            else:
                ilen = L - 2.0; icx = 0.0
            ib = P.build_car_interior(f"Car{i}_int", coll, ilen, seat_row, rows - icx)
            ib.location = (icx, 0, 0)
            ib.parent = ob
            if kind == "end":
                cab = P.build_cab(f"Car{i}_cab", coll, X_origin); cab.parent = ob
                wip = P.build_wiper(f"Car{i}_wiper", coll, X_origin); wip.parent = ob
                dfl = P.build_deflector(f"Car{i}_deflector", coll, X_origin); dfl.parent = ob
                emb = P.build_emblem(f"Car{i}_emblem", coll, X_origin); emb.parent = ob
        # gangway at the rear coupling plane (except last car)
        if i < S.N_CARS:
            g = P.build_gangway(f"Gangway{i}", coll)
            rear_local = (c["centre"] - (c["start"] + L)) if c["kind"] != "end_rev" else None
            g.location = (c["centre"] - (c["start"] + L), 0, 0)
            g.parent = ob
        # bogies
        for k, along in enumerate(c["bogies"]):
            motor = c["motor"]
            bo = bpy.data.objects.new(f"Car{i}_bogie{k}", (bg_m if motor else bg_t).data)
            coll.objects.link(bo); bo.location = (-along, 0, 0)
            res["bogies"].append((bo, i, along))
            for w, dx in enumerate((-S.BOGIE_WHEELBASE / 2, S.BOGIE_WHEELBASE / 2)):
                wo = bpy.data.objects.new(f"Car{i}_b{k}_ws{w}", (ws_m if motor else ws_t).data)
                coll.objects.link(wo); wo.parent = bo
                wo.location = (dx, 0, S.WHEEL_DIAMETER / 2)
                res["wheelsets"].append(wo)
        # pantographs on cars 3 and 6 (roof well)
        if c["panto"]:
            pb, rig = P.build_pantograph(f"Panto{i}", coll)
            pb.parent = ob; pb.location = (0.0 if True else 0, 0, 3.86)
            raised = i == raised_panto_car
            head = (S.CONTACT_WIRE_HEIGHT - 3.86) if raised else 0.62
            arm = P.pose_pantograph(f"Panto{i}", coll, rig, head, pb)
            res["pantos"].append(dict(car=i, base=pb, arm=arm, rig=rig, raised=raised))
    return res
