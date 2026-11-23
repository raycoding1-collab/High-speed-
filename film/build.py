"""Build the complete film scene into one .blend (run with the bpy module or Blender's python).

usage: python -m film.build --assets ASSETS --out scene.blend [--quality preview|final] [--no-veg]
"""
import argparse
import math
import os
import sys
import time

import bpy
import numpy as np

from . import alignment as A
from . import bl, infra, shots as SH, specs as S, train as TR, world as W
from . import materials as M

PADDY_PATH = [None]
QUALITY = {
    "preview": dict(res=(640, 360), samples=16, veg_scale=0.6),
    "still": dict(res=(1920, 1080), samples=96, veg_scale=1.0),
    "final": dict(res=(1920, 1080), samples=48, veg_scale=1.0),         # measured: ~5.7 min/frame on a 4-core runner
    "final4k": dict(res=(3840, 2160), samples=96, veg_scale=1.0),
}


def log(*a):
    print("[build]", *a, flush=True)


def tex_paths(assets):
    t = os.path.join(assets, "tex")
    def p(aid):
        return dict(diff=f"{t}/{aid}/{aid}_diff_2k.jpg", rough=f"{t}/{aid}/{aid}_rough_2k.jpg", nor=f"{t}/{aid}/{aid}_nor_gl_2k.png")
    return dict(concrete=p("concrete_wall_008"), slab=p("precast_concrete_wall"), rock=p("rock_face_03"), cliff=p("marble_cliff_02"),
                forest=p("forest_ground_04"), grass=p("sparse_grass"), mud=p("brown_mud_02"), gravel=p("gravel_floor_02"),
                aerial=p("aerial_grass_rock"))


# ------------------------------------------------------------------ train kinematics
def train_pose(res, s_head, t):
    """World matrices for bogies, cars, wheel angles at head chainage s_head (time t for sway phases)."""
    off = A.track_offset()
    out = {}
    piv = {}
    for bo, ci, along in res["bogies"]:
        O, T, N, U = A.frame(np.array([s_head - along]), off)
        out[bo.name] = (O[0], T[0], N[0], U[0])
        piv.setdefault(ci, []).append((along, O[0], U[0]))
    for car in res["cars"]:
        ci = int(car.name[3:])
        (a1, p1, u1), (a2, p2, u2) = sorted(piv[ci], key=lambda q: q[0])
        P = 0.5 * (p1 + p2)
        T = p1 - p2; T /= np.linalg.norm(T)
        U = 0.5 * (u1 + u2); U -= T * np.dot(U, T); U /= np.linalg.norm(U)
        N = np.cross(U, T)
        # secondary-suspension motion: lateral sway, roll, bounce (mm / tenths of a degree), car-specific phases
        ph = ci * 1.7
        lat = 0.006 * math.sin(2 * math.pi * 1.05 * t + ph) + 0.003 * math.sin(2 * math.pi * 2.7 * t + 2 * ph)
        bounce = 0.004 * math.sin(2 * math.pi * 1.4 * t + 0.5 * ph) + 0.0015 * math.sin(2 * math.pi * 6.1 * t + ph)
        roll = math.radians(0.12) * math.sin(2 * math.pi * 0.9 * t + ph)
        P = P + N * lat + U * (bounce + 0.0)
        c, s_ = math.cos(roll), math.sin(roll)
        N2 = N * c + U * s_; U2 = -N * s_ + U * c
        if car.get("rev"):
            T, N2 = -T, -N2
        out[car.name] = (P, T, N2, U2)
    out["_wheel_angle"] = (s_head - 0.0) / (S.WHEEL_DIAMETER / 2)
    return out


def _mat(P, T, N, U):
    from mathutils import Matrix
    return Matrix(((T[0], N[0], U[0], P[0]), (T[1], N[1], U[1], P[1]), (T[2], N[2], U[2], P[2]), (0, 0, 0, 1)))


def key_object_matrix(ob, M, frame):
    ob.matrix_world = M
    ob.keyframe_insert("location", frame=frame)
    ob.keyframe_insert("rotation_euler", frame=frame)


def animate(scene, res, ocs, shots):
    tl = SH.timeline(shots)
    scene.frame_start = 1; scene.frame_end = tl[-1][2]
    scene.render.fps = SH.FPS
    panto = [p for p in res["pantos"] if p["raised"]][0]
    panto_along = [c for c in res["layout"] if c["index"] == panto["car"]][0]["centre"]
    frames, poses, sheads = [], [], []
    for sh, f0, f1 in tl:
        for f in range(f0, f1 + 1):
            for sub in (0.0, 0.5):
                t = sh.sim((f - f0 + sub) / SH.FPS)
                s_head = sh.s_head(t)
                frames.append(f + sub); sheads.append(s_head)
                poses.append(train_pose(res, s_head, t))
    frames = np.array(frames, np.float32)
    for ob in res["cars"] + [b for b, _, _ in res["bogies"]]:
        mats = np.array([np.array(_mat(*p[ob.name])) for p in poses])
        bl.set_matrix_keys(ob, frames, mats)
    ang = np.array([p["_wheel_angle"] for p in poses])
    for w in res["wheelsets"]:
        w.rotation_mode = "XYZ"
        bl.set_keys(w, "rotation_euler", 1, frames, ang)
    sp = np.array(sheads) - panto_along
    for wob in ocs["contact"] + ocs["messenger"]:
        md = wob.modifiers["uplift"]
        bl.modifier_input(md, "Amplitude").value = 0.12 if wob in ocs["contact"] else 0.05
        inp = bl.modifier_input(md, "Panto S")
        bl.set_keys(wob, inp.path_from_id("value"), 0, frames, sp)


def key_grass_wake(shots):
    """Feed the train head/tail chainage to the vegetation scatter (slipstream tilt) for every frame."""
    frames, heads = [], []
    for sh, f0, f1 in SH.timeline(shots):
        for f in range(f0, f1 + 1):
            for sub in (0.0, 0.5):
                frames.append(f + sub); heads.append(sh.s_head(sh.sim((f - f0 + sub) / SH.FPS)))
    frames = np.array(frames, np.float32); heads = np.array(heads)
    for name, amp in (("VegCuttingGrass", 0.9), ("VegHillShrubs", 0.25), ("VegPaddyGrass", 0.0)):
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        md = ob.modifiers["scatter"]
        bl.modifier_input(md, "Wake").value = amp
        bl.modifier_input(md, "Track Off").value = A.track_offset()
        bl.set_keys(ob, bl.modifier_input(md, "Head S").path_from_id("value"), 0, frames, heads)
        bl.set_keys(ob, bl.modifier_input(md, "Tail S").path_from_id("value"), 0, frames, heads - S.TRAIN_LENGTH)


def _fcurves(action):
    try:
        return list(action.fcurves)
    except AttributeError:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out.extend(cb.fcurves)
        return out


# exposure per shot (stops), like a camera operator opening up for shots inside the shaded cutting
SHOT_EXPOSURE = {"S6": -1.6, "S7": 0.5, "S9": 0.6, "SX": 0.3}


def key_shutter(scene, shots, t_fast=1.0 / 2000.0):
    """Shutter across the speed ramps, like a high-speed camera: 180 deg (1/48 s) at real time, easing to a
    1/2000 s exposure at the slowest playback, so slow-motion frames are razor sharp (train blur 111 m/s x 1/2000 s
    = 5.6 cm instead of 2.3 m).  Exposure interpolates logarithmically with the playback speed; the shutter value
    is in screen frames: frac = exposure x 24 / playback speed."""
    frames, vals = [], []
    for sh, f0, f1 in SH.timeline(shots):
        for f in range(f0, f1 + 1):
            sp = sh.speed(sh.sim((f - f0) / SH.FPS)) if sh.ramp else 1.0
            if sh.ramp:
                lo = sh.ramp[2]
                w = float(np.clip((sp - lo) / (1.0 - lo), 0.0, 1.0))
                expo = math.exp(w * math.log(1.0 / 48.0) + (1 - w) * math.log(t_fast))
                frac = min(0.5, expo * SH.FPS / max(sp, 1e-3))
            else:
                frac = 0.5
            frames.append(f); vals.append(frac)
    for f, v in zip(frames, vals):
        scene.render.motion_blur_shutter = v
        scene.render.keyframe_insert("motion_blur_shutter", frame=f)
    scene.render.motion_blur_shutter = 0.5
    log("  shutter keyed: min", round(min(vals), 3), "frames")


# camera white balance per shot, like a DP setting the camera for the light it shoots in: the cutting shots lie
# in shade lit by the blue morning sky (~8000 K), so they are balanced for shade instead of rendering cold navy
SHOT_WB = {"S7": 7000.0, "S9": 7000.0, "SX": 6800.0}


def key_shot_exposure(scene, shots):
    scene.view_settings.use_white_balance = True
    for sh, f0, f1 in SH.timeline(shots):
        scene.view_settings.exposure = SHOT_EXPOSURE.get(sh.name[:2], 0.0)
        scene.view_settings.keyframe_insert("exposure", frame=f0)
        scene.view_settings.white_balance_temperature = SHOT_WB.get(sh.name[:2], 6500.0)
        scene.view_settings.keyframe_insert("white_balance_temperature", frame=f0)
    s7 = [(sh, f0, f1) for sh, f0, f1 in SH.timeline(shots) if sh.name.startswith("S7")]
    if s7:
        sh, f0, f1 = s7[0]
        for f in range(f0, f1 + 1):
            t = sh.sim((f - f0) / SH.FPS)
            w = min(max((t - 2.55) / 0.35, 0.0), 1.0); w = w * w * (3 - 2 * w)
            scene.view_settings.exposure = SHOT_EXPOSURE["S7"] + 1.3 * w
            scene.view_settings.keyframe_insert("exposure", frame=f)
    # the finale fades to black over its last 1.25 s (per-frame keys, so CONSTANT interpolation is fine)
    sx = [(sh, f0, f1) for sh, f0, f1 in SH.timeline(shots) if sh.name.startswith("SX")]
    if sx:
        sh, f0, f1 = sx[0]
        n = 30
        for k in range(n + 1):
            scene.view_settings.exposure = SHOT_EXPOSURE["SX"] - 10.0 * (k / n) ** 1.6
            scene.view_settings.keyframe_insert("exposure", frame=f1 - n + k)
    if scene.animation_data and scene.animation_data.action:
        for fc in _fcurves(scene.animation_data.action):
            if fc.data_path in ("view_settings.exposure", "view_settings.white_balance_temperature"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


FOCUS = {"S1": 760.0, "S2": 45.0, "S3": "Car3", "S4": "Car4", "S5": 120.0, "S6": 9.5, "S7": 4.2, "S8": "Car1", "S9": 60.0, "SX": 80.0}


def make_cameras(scene, shots, res):
    tl = SH.timeline(shots)
    cams = []
    for sh, f0, f1 in tl:
        cd = bpy.data.cameras.new(sh.name)
        cd.lens = sh.lens; cd.sensor_width = 36.0
        cd.clip_start = 0.05; cd.clip_end = 20000
        ob = bpy.data.objects.new("CAM_" + sh.name, cd)
        scene.collection.objects.link(ob)
        rng = np.random.default_rng(abs(hash(sh.name)) % 2**32)
        ob.rotation_mode = "XYZ"
        if sh.attach:
            parent = bpy.data.objects[sh.attach[0]]
            ob.parent = parent
            ob.location = sh.attach[1]
            if sh.name.startswith("S5"):
                ob.rotation_euler = (math.radians(86.5), 0, math.radians(-90))     # look forward (+x), slightly down
            else:
                ob.rotation_euler = (math.radians(79.0), 0, math.radians(97))      # look back (-x) and across at the pantograph
            fr = np.arange(f0, f1 + 2, dtype=np.float32)
            j = rng.normal(0, sh.shake, (len(fr), 3))
            for k in range(3):
                bl.set_keys(ob, "location", k, fr, sh.attach[1][k] + j[:, k] * (0.2 if k < 2 else 1.0))
        else:
            fr, locs, rots = [], [], []
            prev = None
            from mathutils import Vector
            for f in range(f0, f1 + 2):
                for sub in (0.0, 0.5):
                    t = sh.sim((f - f0 + sub) / SH.FPS)
                    pos = sh.cam_fn(t)
                    aim = sh.aim_fn(t) if sh.aim_fn is not None else _train_aim(sh, t)
                    d = Vector(tuple(aim)) - Vector(tuple(pos))
                    q = d.to_track_quat("-Z", "Y")
                    e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
                    prev = e.copy()
                    ex, ez = e.x, e.z
                    if sh.shake:
                        ex += rng.normal(0, sh.shake) * 0.002; ez += rng.normal(0, sh.shake) * 0.002
                    if sh.jolt:
                        if f == f0 and sub == 0.0:
                            s_cam = float(A.nearest_s(np.array([pos[0]]), np.array([pos[1]]))[0][0])
                        dyaw, dpitch, dlat = SH.pressure_jolt(sh, t, s_cam)
                        ex += dpitch; ez += dyaw
                        th = A.heading(s_cam)
                        pos = np.asarray(pos, float) + dlat * np.array([-math.sin(th), math.cos(th), 0.0])
                    fr.append(f + sub); locs.append(tuple(pos)); rots.append((ex, e.y, ez))
            fr = np.array(fr, np.float32); locs = np.array(locs); rots = np.array(rots)
            for k in range(3):
                bl.set_keys(ob, "location", k, fr, locs[:, k]); bl.set_keys(ob, "rotation_euler", k, fr, rots[:, k])
            ob.location = tuple(locs[0])
        if sh.fstop:
            cd.dof.use_dof = True; cd.dof.aperture_fstop = sh.fstop
            foc = FOCUS.get(sh.name[:2], 50.0)
            if isinstance(foc, str):
                cd.dof.focus_object = bpy.data.objects[foc]       # focus puller follows the lead car
            else:
                cd.dof.focus_distance = foc
        m = scene.timeline_markers.new(sh.name, frame=f0); m.camera = ob
        cams.append(ob)
    scene.camera = cams[0]
    return cams


def _train_aim(sh, t):
    """Default aim: operator following the train (drone/heli shots)."""
    off = A.track_offset()
    if sh.name.startswith("S3"):
        # drone: looks ahead-left first, then the operator pans to follow the nose as it passes and recedes
        cam = sh.cam_fn(t)
        nose = A.frame(np.array([sh.s_head(t)]), off)[0][0] + np.array([0, 0, 2.0])
        ahead = A.frame(np.array([1650 + 26 * t + 120]), off)[0][0] + np.array([0, 0, 1.0])
        w = np.clip((sh.s_head(t) - (1650 + 26 * t) + 60) / 120, 0, 1)
        return ahead * (1 - w) + nose * w
    if sh.name.startswith("S4"):
        along = 105 - 13 * t                                  # operator keeps the front half framed as it draws ahead
        return A.frame(np.array([sh.s_head(t) - along]), off)[0][0] + np.array([0, 0, 2.0])
    return A.frame(np.array([sh.s_head(t)]), off)[0][0]


def build(assets, out, quality="preview", veg=True, texdir=None, only_shots=None, pack=False):
    t0 = time.time()
    q = QUALITY[quality]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bl.set_cycles(sc, samples=q["samples"])
    sc.render.resolution_x, sc.render.resolution_y = q["res"]
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5           # 180-degree shutter
    sc.render.motion_blur_position = "START"
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"    # v3: filmic contrast, deeper blacks, same highlight roll-off
    except TypeError:
        pass
    texdir = texdir or os.path.join(os.path.dirname(out), "textures")
    os.makedirs(texdir, exist_ok=True)
    tex = tex_paths(assets)

    log("sky + haze")
    # v3: late-afternoon golden light, sun 19 deg high in the WSW (measured from the HDRI: u 0.6004, 19.1 deg),
    # turned so it shines straight down the cutting onto the portal and lights the eastbound views from behind.
    # (v1/v2 used a 4.6 deg sunrise: the cutting, portal and finale sat in deep blue shade.)
    sun, sun_dir = W.build_sky(sc, os.path.join(assets, "hdri", "qwantani_late_afternoon_puresky_8k.hdr"),
                               sun_azimuth_deg=260.3, hdri_sun=(0.6004, 19.09), sun_color=(1.0, 0.80, 0.58),
                               sun_energy=7.0)
    cw = bl.collection("World")
    from . import post
    post.setup(sc, haze_on=False)                         # bloom + grade only; atmosphere is analytic fog in the shaders

    log("terrain")
    pimg_path = os.path.join(texdir, "paddies.png")
    if not os.path.exists(pimg_path):
        W.paddy_image(pimg_path)
    pimg = bpy.data.images.load(pimg_path)
    PADDY_PATH[0] = pimg_path
    mats = dict(terrain=W.terrain_material(tex, pimg), karst=W.karst_material(tex))
    far, strip = W.build_terrain(cw, mats)
    towers = W.build_towers(cw, mats)
    W.build_river(cw)
    W.build_roads(cw, tex, W.hamlet_sites())

    log("infrastructure")
    ci = bl.collection("Railway")
    infra.build_viaduct(ci, tex)
    infra.build_piers(ci, tex, lambda x, y: float(W.natural_ground(np.array([x]), np.array([y]))[0]))
    infra.build_track(ci, tex, A.S_MIN + 20, A.TUNNEL_S + 280)
    infra.build_noise_barriers(ci, 2350.0, 3150.0)
    ocs = infra.build_ocs(ci, A.S_MIN + 30, A.TUNNEL_S + 270)
    build_tunnel(ci, tex)
    infra.build_furniture(ci, lambda x, y: float(W.natural_ground(np.array([x]), np.array([y]))[0]))
    infra.build_cutting_details(ci, tex)
    infra.build_signs(ci, texdir)
    W.build_village(cw, assets)
    hr = np.random.default_rng(31)
    for i, (hs, hl, hx, hy) in enumerate(W.hamlet_sites()):
        featured = (hs, hl) in W.FEATURED_HAMLETS              # in-shot hamlets: bigger, so they read in the wide shots
        W.build_village(cw, assets, n=int(hr.integers(15, 23)) if featured else int(hr.integers(6, 15)), centre_s=hs, lat=hl,
                        seed=40 + i, spread_s=95.0 if featured else 75.0, spread_l=(-75.0, 75.0) if featured else (-60.0, 60.0),
                        name=f"Hamlet{i}", hamlet=True)
    from . import props as PR
    hs_all = W.hamlet_sites()
    hs_all = [h for h in hs_all if (h[0], h[1]) in W.FEATURED_HAMLETS] + [h for h in hs_all if (h[0], h[1]) not in W.FEATURED_HAMLETS]
    PR.build_props(cw, bl.collection("VegSources"), hs_all)         # featured (in-shot) hamlets get their props first

    log("train")
    ct = bl.collection("Train")
    res = TR.build_train(texdir, ct)
    for car in res["cars"]:
        if car.name == "Car8":
            car["rev"] = 1
    if veg:
        log("vegetation")
        build_vegetation(cw, assets, towers, q["veg_scale"])
    log("wake effects")
    from . import wake
    wake.build_wake_effects(bl.collection("Wake"), lambda x, y: W.ground(x, y))
    log("cameras + animation")
    shots = SH.build_shots()
    animate(sc, res, ocs, shots)
    key_grass_wake(shots)
    make_cameras(sc, shots, res)
    key_shot_exposure(sc, shots)
    key_shutter(sc, shots)
    for ob in res["wheelsets"]:
        try:
            ob.cycles.motion_steps = 6
        except Exception:
            pass
    from . import fog
    log("atmospheric fog in", fog.apply(sc, sun_dir, s_mist=0.0004, s_haze=0.00006, sun_col=(1.0, 0.82, 0.60)),
        "materials")   # v3: afternoon - thin ground haze, clearer mid-distance
    sc.render.film_transparent = False
    bpy.ops.file.make_paths_relative() if False else None
    if pack:
        log("packing all images into the .blend")
        bpy.ops.file.pack_all()
    log("saving", out)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out), compress=True)
    log(f"done in {time.time() - t0:.0f} s")
    return sc


def build_tunnel(coll, tex):
    """Double-track HSR tunnel (horseshoe, ~100 m2) with an oblique 'hat-brim' portal."""
    con = infra.concrete("PortalConcrete", tex.get("concrete"), (0.88, 0.87, 0.84), 0.22)
    R = 6.8
    prof_out, prof_in = [], []
    for k in range(25):
        a = math.radians(-10 + 200 * k / 24)
        prof_out.append(((R + 0.7) * math.cos(a), 1.2 + (R + 0.7) * math.sin(a)))
        prof_in.append((R * math.cos(a), 1.2 + R * math.sin(a)))
    prof = prof_out + prof_in[::-1]
    o = infra.sweep("TunnelLining", [(y, z - 0.9) for y, z in prof], A.TUNNEL_S - 10, A.TUNNEL_S + 290, 2.0, coll=coll, mat=con, closed=True)
    # oblique cut: push the lower front vertices back into the hill (hat-brim portal ~ 1:1.5 slope)
    me = o.data
    s0 = A.TUNNEL_S - 10
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    s_v, lat_v = A.nearest_s(co[:, 0], co[:, 1])
    front = s_v < s0 + 1.0
    h = co[:, 2] - (A.RAIL_Z - 0.9)
    shift = np.clip(10.0 - h * 0.75, 0, 10.0)
    th = A.heading(s_v)
    co[front, 0] += np.cos(th[front]) * shift[front]; co[front, 1] += np.sin(th[front]) * shift[front]
    me.vertices.foreach_set("co", co.ravel()); me.update()
    # (v3) portal dressing for the finale: granite name plaque on the hood crown (gold characters), a concrete
    # collar band along the oblique edge, and the row of wall lamps every 25 m fading into the dark
    from .trainparts import box, setmat
    import os as _os
    font = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    plq = infra._sign_material("TunnelName", "桂 岭 隧 道" if _os.path.exists(font) else "GUILING TUNNEL", (1024, 256),
                               bg=(22, 22, 24), fg=(214, 176, 92), font=font if _os.path.exists(font) else None, border=True,
                               texdir=_os.path.join(_os.path.dirname(tex["concrete"]["diff"]), "..", "..", "signs"))
    O, T, N, U = infra._frames_flat(np.array([A.TUNNEL_S - 5.2]))
    O, T, N = O[0], T[0], N[0]
    top = (A.RAIL_Z - 0.9) + 1.2 + R + 0.7
    pl = infra._plate("TunnelPlaque", 4.6, 1.15, plq, coll)
    pl.location = tuple(O + np.array([0, 0, top - O[2] + 0.35]))
    pl.rotation_euler = (0, 0, math.atan2(T[1], T[0]) - math.pi / 2)        # faces the approaching train
    base_m = infra.concrete("PlaqueBase", tex.get("concrete"), (0.7, 0.69, 0.66), 0.3)
    b = setmat(box("TunnelPlaqueBase", (0.5, 5.0, 0.4), (0, 0, 0), coll), base_m)
    b.location = tuple(O + np.array([0, 0, top - O[2] + 0.15])); b.rotation_euler = (0, 0, math.atan2(T[1], T[0]))
    lamp = M.principled("TunnelLamp", (1.0, 0.85, 0.6), 0.0, 0.3, emission=(1.0, 0.82, 0.55), estrength=40.0)
    for s_l in np.arange(A.TUNNEL_S + 12, A.TUNNEL_S + 290, 25):
        Ol, Tl, Nl, Ul = infra._frames_flat(np.array([s_l]))
        for side in (-1, 1):
            p_ = Ol[0] + Nl[0] * side * (R - 0.25) + np.array([0, 0, 4.2])
            lb = setmat(box(f"TunLamp{int(s_l)}_{side}", (0.6, 0.12, 0.12), tuple(p_), coll), lamp)
            lb.rotation_euler = (0, 0, math.atan2(Tl[0][1], Tl[0][0]))
    # interior lights every 50 m (dim)
    for s in np.arange(A.TUNNEL_S + 40, A.TUNNEL_S + 290, 50):
        ld = bpy.data.lights.new(f"TunLight{s:.0f}", "POINT"); ld.energy = 60; ld.color = (1.0, 0.85, 0.6); ld.shadow_soft_size = 0.2
        lo = bpy.data.objects.new(f"TunLight{s:.0f}", ld); coll.objects.link(lo)
        p = infra._frames_flat(np.array([s]))[0][0]
        lo.location = (p[0], p[1], p[2] + 5.5)
    return o


def _sightline_mask(p, clear=9.0):
    """False for points close to any static camera's line of sight (and around the camera)."""
    keep = np.ones(len(p), bool)
    for sh in SH.build_shots():
        if sh.attach or sh.cam_fn is None or sh.aim_fn is None:
            continue
        # (v3) the whole shot, not just its first frame: the S8 600 mm lens follows the train for 9 s and a single
        # fir on the rim in front of it blacked out the frame; long lenses keep their full line of sight clear
        for t in np.linspace(0.0, sh.sim_dur, 13 if sh.lens >= 100 else 5):     # long lenses pan: sample densely
            c = np.asarray(sh.cam_fn(t)); a = np.asarray(sh.aim_fn(t))
            d = a - c; L = np.linalg.norm(d); d = d / L
            long = sh.lens >= 100
            # the points are tree BASES: test the whole trunk-to-crown height (0-18 m), a fir standing below the
            # line of sight reaches up into it (that is what kept blocking the 600 mm frame)
            beyond = ((p - c) @ d) > L + 3.0                     # behind the target: cannot block the view of it
            for hz in (0.0, 6.0, 12.0, 18.0):
                v = p + np.array([0.0, 0.0, hz]) - c
                tt = np.clip(v @ d, 0, L if long else min(L, 400))
                dist = np.linalg.norm(v - tt[:, None] * d[None, :], axis=1)
                keep &= beyond | (dist > np.maximum(14.0 if long else clear, 0.04 * tt))
            keep &= np.linalg.norm(p - c, axis=1) > (45.0 if long else 20.0)   # nothing growing right at the camera
    return keep


def build_vegetation(coll, assets, towers, scale=1.0):
    rng = np.random.default_rng(11)
    models, src = W.load_models(assets, {
        "tree1": ("island_tree_01", "island_tree_01_LOD0", 9.0), "tree2": ("island_tree_02", "island_tree_02_LOD0", 7.5),
        "tree3": ("island_tree_03", "island_tree_03_LOD0", 6.5), "tree4": ("tree_small_02", "tree_small_02_LOD1", 5.5),
        "shrub1": ("shrub_01", "shrub_01_h_LOD0", 1.6), "shrub2": ("shrub_02", "shrub_02_c_LOD0", 1.3), "shrub4": ("shrub_04", "shrub_04_d_LOD0", 1.8),
        "fern": ("fern_02", "fern_02_b", 0.7), "grass1": ("grass_medium_01", "grass_medium_01_large_a_LOD0", 0.45),
        "grass2": ("grass_medium_02", "grass_medium_02_d", 0.5), "grass3": ("grass_medium_01", "grass_medium_01_tall_a_LOD0", 0.55),
        "weed": ("weed_plant_02", "weed_plant_02_a_LOD0", 0.5),
        # v3: Masson pine and Chinese fir, the plantation conifers of the Guangxi hills, for the slopes near cameras
        "pine": ("pine_tree_01", "pine_tree_01_a_LOD2", 15.0), "fir": ("fir_tree_01", "fir_tree_01_a_LOD1", 14.0),
    })
    for k, o in models.items():
        log("  model", k, o.name, len(o.data.polygons))
    trees = [models[k] for k in ("tree1", "tree2", "tree3", "tree4") if k in models]
    conifers = [models[k] for k in ("pine", "fir") if k in models]
    hill_species = (conifers * 3 + trees[3:4]) if conifers else trees            # 3:3:1 pine : fir : broadleaf
    shrubs = [models[k] for k in ("shrub1", "shrub2", "shrub4", "fern") if k in models]
    grasses = [models[k] for k in ("grass1", "grass2", "grass3", "weed") if k in models]
    # crowns (mid-field)
    crowns = [W.crown_proxy(f"Crown{i}", src, i) for i in range(8)]
    bamboo = [W.bamboo_proxy(f"Bamboo{i}", src, 100 + i) for i in range(4)]
    for c in crowns + bamboo:
        c.hide_render = True
    # towers: crowns on ledges / gentle slopes
    def tower_filter(p, n):
        # dense subtropical scrub forest everywhere it can cling; bare limestone only on the steepest faces
        keep_p = np.clip((n[:, 2] - 0.18) / 0.35, 0, 1) * 0.95
        return rng.random(len(p)) < keep_p
    pts = W.sample_surface(towers, 0.023 * scale, rng, tower_filter, max_n=900000)
    log("  tower crowns", len(pts))
    W.scatter_tree("VegTowers", coll, pts, crowns, (2.8, 6.5), 1, wind=0.0)
    # hill massif & corridor slopes near the cutting / portal: real trees + shrubs + grass
    strip = bpy.data.objects["TerrainCorridor"]
    far = bpy.data.objects["TerrainFar"]
    def slope_near(lo, hi, lat_min, clear=True):
        def f(p, n):
            s, lat = A.nearest_s(p[:, 0], p[:, 1])
            face = (s > A.TUNNEL_S - 6.0) & (p[:, 2] > A.RAIL_Z + 9.0)          # hill face above the portal
            k = (s > lo) & (s < hi) & ((np.abs(lat) > lat_min) | face) & (n[:, 2] > 0.42)
            return k & _sightline_mask(p, 6.0) if clear else k
        return f
    pts = W.sample_surface([strip], 0.035 * scale, rng, slope_near(3700, 4700, 30), max_n=40000)
    log("  hill trees", len(pts))
    W.scatter_tree("VegHillTrees", coll, pts, hill_species, (0.6, 1.1), 2, wind=0.03)
    pts = W.sample_surface([strip], 0.30 * scale, rng, slope_near(3750, 4660, 8.5), max_n=90000)
    W.scatter_tree("VegHillShrubs", coll, pts, shrubs, (0.7, 1.5), 3, wind=0.06)
    def cut_grass(p, n):
        k = slope_near(3900, 4520, 6.6, clear=False)(p, n)
        s_, lat_ = A.nearest_s(p[:, 0], p[:, 1])
        return k & ~infra.on_lattice(s_, np.abs(lat_)) & ~((np.abs(lat_) > 5.6) & (np.abs(lat_) < 6.9))   # not on beams / in drains
    pts = W.sample_surface([strip], 5.5 * scale, rng, cut_grass, max_n=320000)
    log("  cutting grass", len(pts))
    # close to the S7 trackside camera the hydroseeded pockets are thick with grass (the camera is 3 m from the slope)
    c7 = np.asarray([sh for sh in SH.build_shots() if sh.name.startswith("S7")][0].cam_fn(0.0), float)
    def near7(p, n):
        return cut_grass(p, n) & (np.hypot(p[:, 0] - c7[0], p[:, 1] - c7[1]) < 45.0)
    p7 = W.sample_surface([strip], 32.0 * scale, rng, near7, max_n=120000,
                          tri_filter=lambda c: np.hypot(c[:, 0] - c7[0], c[:, 1] - c7[1]) < 48.0)
    if len(pts):
        pts = np.concatenate([pts, p7])
    log("  + dense grass at the S7 camera", len(p7))
    W.scatter_tree("VegCuttingGrass", coll, pts, grasses, (0.8, 1.6), 4, wind=0.10)
    # valley floor: dense green tree clumps (copses, river lines, hamlet groves) and a few lone trees, like real
    # southern-Chinese farmland. (Evenly scattered single Poly Haven "island" trees read as savanna, so not used here.)
    vp = W.valley_tree_points(rng, W.hamlet_sites(), scale)
    if W.ROADS:                                     # tree lines along the village roads (gaps, both sides)
        rp = []
        for (xs, ys) in W.ROADS:
            dx = np.gradient(xs); dy = np.gradient(ys); nn = np.hypot(dx, dy) + 1e-9
            for sgn in (-1, 1):
                k = rng.random(len(xs)) < 0.28
                off = sgn * rng.uniform(5.0, 7.0, len(xs))
                rp.append(np.column_stack([xs[k] - dy[k] / nn[k] * off[k], ys[k] + dx[k] / nn[k] * off[k]]))
        rp = np.concatenate(rp)
        rp = rp[W._clear_of_features(rp[:, 0], rp[:, 1], 4.0, river=21.0)]
        _, l_ = A.nearest_s(rp[:, 0], rp[:, 1])
        rp = rp[np.abs(l_) > 22.0]                  # roads pass under the viaduct: no trees between the piers
        vp = np.concatenate([vp, np.column_stack([rp, W.ground(rp[:, 0], rp[:, 1])])])
        log("  road-side trees", len(rp))
    s5 = [x for x in SH.build_shots() if x.name.startswith("S5")][0]
    ss5 = rng.uniform(s5.s0 - 50, s5.s0 + S.SPEED * s5.sim_dur + 250, 2600)
    ll5 = rng.choice([-1, 1], len(ss5)) * rng.uniform(24, 70, len(ss5))
    th5 = A.heading(ss5); x5, y5 = A.xy(ss5)
    gx5, gy5 = x5 - np.sin(th5) * ll5, y5 + np.cos(th5) * ll5
    k5 = W._clear_of_features(gx5, gy5, 4.0, river=21.0) & (rng.random(len(ss5)) < 0.75)
    side5 = np.column_stack([gx5[k5], gy5[k5], W.ground(gx5[k5], gy5[k5])])
    side5 = side5[_sightline_mask(side5, 9.0)]                     # keep the 600 mm and other static views clear
    log("  trees beside the viaduct on the cab run", len(side5))
    vp = vp[_sightline_mask(vp, 9.0)]
    from . import village as V
    if V.SITES:                                     # no tree grows through a house (houses are <= 14 m across)
        hs = np.array([(q[0], q[1]) for q in V.SITES])
        near = np.zeros(len(vp), bool)
        for hx, hy in hs:
            near |= np.hypot(vp[:, 0] - hx, vp[:, 1] - hy) < 12.0
        log("  valley trees removed at houses", int(near.sum()))
        vp = vp[~near]
    from . import props as PR
    def clear_of_props(p):
        k = np.ones(len(p), bool)
        for (kx, ky, kr) in PR.KEEPOUT:
            k &= np.hypot(p[:, 0] - kx, p[:, 1] - ky) > kr
        return k
    vp = vp[clear_of_props(vp)]
    # crown proxies are mid-distance trees: keep them >= 300 m from ground-level cameras (S2 tripod in the paddies),
    # where their simplified shape would be seen up close
    for sh in SH.build_shots():
        if sh.attach or sh.cam_fn is None:
            continue
        c = np.asarray(sh.cam_fn(0.0), float)
        if c[2] - W.natural_ground(np.array([c[0]]), np.array([c[1]]))[0] < 6.0:
            vp = vp[np.hypot(vp[:, 0] - c[0], vp[:, 1] - c[1]) > 300.0]
    # bamboo along the river and in the groves round the hamlets, broadleaf copses elsewhere
    river_d = np.abs(vp[:, 0] - W.RIVER_X(vp[:, 1] + 500.0))
    near_h = np.zeros(len(vp), bool)
    for (hs_, hl_, hx_, hy_) in W.hamlet_sites():
        near_h |= np.hypot(vp[:, 0] - hx_, vp[:, 1] - hy_) < 90.0
    is_bamboo = (river_d < 45.0) | (near_h & (rng.random(len(vp)) < 0.45))
    log("  valley trees (crown clumps)", int((~is_bamboo).sum()), "bamboo clumps", int(is_bamboo.sum()))
    # trees the S1 drone flies over (within 230 m of its path) are real broadleaf models, not crown proxies
    path = []
    for sh_ in SH.build_shots():
        if sh_.name[:2] in ("S1", "S3", "S4") and sh_.cam_fn is not None:
            reach = 230.0 if sh_.name.startswith("S1") else 650.0
            for t in np.linspace(0, sh_.sim_dur, 30):
                c_ = np.asarray(sh_.cam_fn(t)); path.append((c_[0], c_[1], reach))
    path = np.array(path)
    d1 = np.hypot(vp[:, 0, None] - path[None, :, 0], vp[:, 1, None] - path[None, :, 1]) - path[None, :, 2]
    near1 = (np.min(d1, axis=1) < 0) & ~is_bamboo
    if "tree4" in models and near1.any():
        W.scatter_tree("VegValleyNearS1", coll, vp[near1], [models["tree4"]], (1.3, 1.9), 16, wind=0.03)
    log("  real broadleaf trees near the S1/S3/S4 cameras", int(near1.sum()))
    W.scatter_tree("VegValleyCopses", coll, vp[~is_bamboo & ~near1], crowns, (2.2, 5.5), 6, wind=0.0)
    if "tree4" in models and len(side5):
        side5 = side5[clear_of_props(side5)]
        W.scatter_tree("VegViaductSide", coll, side5, [models["tree4"]] + conifers[1:2], (1.2, 1.9), 17, wind=0.03)
    W.scatter_tree("VegValleyBamboo", coll, vp[is_bamboo], bamboo, (3.2, 4.6), 7, wind=0.0)
    # the hill massif around the cutting: forest its broad flanks (seen from the cab and the long lens) where
    # the real-tree corridor strip ends; keep 150 m from the low cameras that look at it close up
    def massif(p, n):
        s_, lat_ = A.nearest_s(p[:, 0], p[:, 1])
        k = (n[:, 2] > 0.45) & (W.natural_ground(p[:, 0], p[:, 1]) > 7.0)
        # the line itself: no trees on the cutting slopes / track (but the hill face above the portal is forested)
        k &= (np.abs(lat_) > 28.0) | ((s_ > A.TUNNEL_S - 6.0) & (p[:, 2] > A.RAIL_Z + 9.0))
        return k & _sightline_mask(p, 6.0)
    # only the hills around the cutting / tunnel that the cameras actually see (the far grid also holds the distant
    # ranges: spreading the budget over all of them left the near massif bare)
    ax, ay = A.xy(np.arange(2600.0, A.S_MAX, 100.0))
    def massif_tris(c):
        d = np.min(np.hypot(c[:, 0, None] - ax[None, :], c[:, 1, None] - ay[None, :]), axis=1)
        return (W.natural_ground(c[:, 0], c[:, 1]) > 7.0) & (d < 1400.0)
    # (memory: every instance costs ~15-20 KB in the render; 85k far crowns still cover the hills seen at 1-3 km)
    pts = W.sample_surface([far], 0.014 * scale, rng, massif, max_n=100000, tri_filter=massif_tris)
    pts = pts[clear_of_props(pts)] if len(pts) else pts
    # real conifers where the cameras are close (within 500 m of the cutting / portal: cab, trackside, 600 mm,
    # portal), simplified crowns beyond
    cx_, cy_ = A.xy(np.arange(3600.0, A.TUNNEL_S + 60, 40.0))
    dmin = np.min(np.hypot(pts[:, 0, None] - cx_[None, :], pts[:, 1, None] - cy_[None, :]), axis=1) if len(pts) else np.zeros(0)
    near = dmin < 500.0
    # simplified crowns never stand within 150 m of the low cutting cameras (real conifers may)
    for sh in SH.build_shots():
        if sh.name[:2] in ("S7", "S9", "SX") and sh.cam_fn is not None and len(pts):
            c = np.asarray(sh.cam_fn(0.0), float)
            close = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < 150.0
            pts, near = pts[~close | near], near[~close | near]
    log("  massif crowns", int((~near).sum()), "real conifers near the cameras", int(near.sum()))
    W.scatter_tree("VegMassif", coll, pts[~near], crowns, (2.6, 5.8), 8, wind=0.0)
    if conifers and near.any():
        W.scatter_tree("VegMassifConifers", coll, pts[near], hill_species, (0.6, 1.15), 9, wind=0.0)
    # grass + weeds right around the S2 camera (paddy edge)
    c2 = [s for s in SH.build_shots() if s.name.startswith("S2")][0].cam_fn(0)
    def near_c2(p, n):
        return np.hypot(p[:, 0] - c2[0], p[:, 1] - c2[1]) < 60
    # S2 foreground (v3): a grassy field bund runs diagonally away from the tripod; beyond it (right) the
    # neighbouring plot is already harvested - rice stubble rows and straw stacks - like the staggered harvest of
    # double-cropped Guangxi paddies; framing trees stand on the bunds at the frame edges
    sh2 = [s_ for s_ in SH.build_shots() if s_.name.startswith("S2")][0]
    a2 = np.asarray(sh2.aim_fn(0.0), float)
    bearing = math.atan2(a2[1] - c2[1], a2[0] - c2[0])
    fwd = np.array([math.cos(bearing), math.sin(bearing)]); lft = np.array([-fwd[1], fwd[0]])
    def to_xy(f_, l_):
        return c2[0] + fwd[0] * f_ + lft[0] * l_, c2[1] + fwd[1] * f_ + lft[1] * l_
    b0, b1 = np.array([6.0, -15.0]), np.array([36.0, 14.0])            # bund end points (forward, left) in metres
    def side(X, Y):                                                    # signed distance to the bund line (+ = right/far)
        f_ = (X - c2[0]) * fwd[0] + (Y - c2[1]) * fwd[1]; l_ = (X - c2[0]) * lft[0] + (Y - c2[1]) * lft[1]
        d = b1 - b0; nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        return (f_ - b0[0]) * nrm[0] + (l_ - b0[1]) * nrm[1]
    W.scatter_rice(coll, src, c2, 42.0 if scale >= 1 else 25.0, PADDY_PATH[0], keep_fn=lambda X, Y: side(X, Y) < -0.6)
    tt = np.linspace(0, 1, 40)
    bx, by = to_xy(b0[0] + (b1[0] - b0[0]) * tt, b0[1] + (b1[1] - b0[1]) * tt)
    from . import materials as M_
    bund_m = M_.principled("FieldBund", (0.075, 0.08, 0.035), 0.0, 0.95)
    W._ribbon("S2Bund", coll, np.asarray(bx), np.asarray(by), [(-0.65, -0.12), (-0.25, 0.30), (0.25, 0.30), (0.65, -0.12)],
              bund_m, W.natural_ground)
    gb = []
    for k in range(900):
        u = rng.random(); w_ = rng.uniform(-0.45, 0.45)
        f_ = b0[0] + (b1[0] - b0[0]) * u; l_ = b0[1] + (b1[1] - b0[1]) * u
        d = b1 - b0; nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        x_, y_ = to_xy(f_ + nrm[0] * w_, l_ + nrm[1] * w_)
        gb.append((x_, y_, float(W.natural_ground(np.array([x_]), np.array([y_]))[0]) + 0.28))
    if grasses:
        W.scatter_tree("VegS2Bund", coll, np.array(gb), grasses, (0.9, 1.5), 13, wind=0.08)
    # harvested plot: stubble rows (short rice clumps) and straw stacks
    clumps = [o for o in bpy.data.objects if o.name.startswith("Rice") and o.type == "MESH"][:4]
    U, Vv = np.meshgrid(np.arange(-40, 40, 0.25), np.arange(-40, 40, 0.30))
    SX, SY = c2[0] + U.ravel(), c2[1] + Vv.ravel()
    sd = side(SX, SY); fdist = (SX - c2[0]) * fwd[0] + (SY - c2[1]) * fwd[1]
    ks = (sd > 0.6) & (fdist > 2) & (np.hypot(SX - c2[0], SY - c2[1]) < 40) & (rng.random(len(SX)) < 0.55)
    if clumps and ks.any():
        SZ = W.natural_ground(SX[ks], SY[ks]) - 0.02
        W.scatter_tree("VegS2Stubble", coll, np.column_stack([SX[ks], SY[ks], SZ]), clumps, (0.18, 0.26), 14, wind=0.0)
    stacks = []          # (the smooth-cone straw stacks read as tents this close to the lens: not used in S2)
    if stacks:
        sp = []
        for f_, l_ in ((14.0, -8.0), (17.5, -11.5), (21.0, -6.5), (26.0, -13.0)):
            x_, y_ = to_xy(f_, l_)
            sp.append((x_, y_, float(W.natural_ground(np.array([x_]), np.array([y_]))[0]) - 0.05))
        W.scatter_tree("PropS2Stacks", coll, np.array(sp), stacks, (0.95, 1.1), 15, wind=0.0)
    fp = []
    for ang, dist in ((-33.0, 22.0), (-26.0, 34.0), (33.0, 24.0), (40.0, 13.0)):     # + a near tree on the left edge
        a = bearing + math.radians(ang)
        x, y = c2[0] + dist * math.cos(a), c2[1] + dist * math.sin(a)
        fp.append((x, y, float(W.natural_ground(np.array([x]), np.array([y]))[0])))
    if trees:
        W.scatter_tree("VegS2Frame", coll, np.array(fp), trees[:3], (1.35, 1.6), 12, wind=0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default="preview"); ap.add_argument("--no-veg", action="store_true")
    ap.add_argument("--texdir", default=None); ap.add_argument("--pack", action="store_true")
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    if a.samples:
        QUALITY[a.quality]["samples"] = a.samples
    build(a.assets, a.out, a.quality, not a.no_veg, a.texdir, pack=a.pack)


if __name__ == "__main__":
    main()
