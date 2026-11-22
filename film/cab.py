"""Driver's cab console for the S5 cab view (car-local coordinates: +X towards the nose, z above rail).

What the 24 mm eye-level camera (X = nose - 5.85 m, z = 2.88 m) sees below the windscreen, like a real CR
cab: a dark textured moulded desk rising from the windscreen base, a row of defroster slots along the glass,
an instrument hood with three tilted driver displays (the CTCS-3 DMI with the 400 km/h speed dial in the middle,
traction/brake data and a line-ahead map either side) set in bezels, a strip of illuminated push buttons and
two rotary switches, and the traction/brake controller handle at the driver's left hand.
The display images are drawn with PIL at build time (no external assets).
"""
import math
import os

import bpy
import numpy as np

from . import materials as M
from .village import MB


def _font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def dmi_images(texdir):
    """Three 1024x640 display images: speed DMI (centre), traction/brake (left), line-ahead map (right)."""
    from PIL import Image, ImageDraw
    os.makedirs(texdir, exist_ok=True)
    W, H = 1024, 640
    bg = (10, 14, 24)
    paths = {}
    # --- centre: CTCS-3 DMI, speed dial
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    cx, cy, R = 330, 330, 260
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(70, 80, 95), width=4)
    vmax = 450.0
    def ang(v):
        return math.radians(-225 + 270 * v / vmax)          # 0 km/h at lower left, 450 at lower right
    for v in range(0, 451, 10):
        a = ang(v); r0 = R - (34 if v % 50 == 0 else 18)
        d.line((cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + (R - 6) * math.cos(a), cy + (R - 6) * math.sin(a)),
               fill=(225, 228, 232), width=5 if v % 50 == 0 else 2)
        if v % 50 == 0:
            tx, ty = cx + (R - 70) * math.cos(a), cy + (R - 70) * math.sin(a)
            f = _font(30); t = str(v); bb = d.textbbox((0, 0), t, font=f)
            d.text((tx - (bb[2] - bb[0]) / 2, ty - (bb[3] - bb[1]) / 2 - 6), t, fill=(225, 228, 232), font=f)
    # permitted-speed hook (white -> 400, then the 420 km/h ceiling in orange)
    for v0, v1, col in ((0, 400, (200, 200, 205)), (400, 420, (240, 150, 30))):
        a0, a1 = math.degrees(ang(v0)), math.degrees(ang(v1))
        d.arc((cx - R + 2, cy - R + 2, cx + R - 2, cy + R - 2), a0, a1, fill=col, width=12)
    a = ang(400.0)
    d.polygon([(cx + (R - 40) * math.cos(a), cy + (R - 40) * math.sin(a)),
               (cx + 14 * math.cos(a + 1.57), cy + 14 * math.sin(a + 1.57)),
               (cx + 14 * math.cos(a - 1.57), cy + 14 * math.sin(a - 1.57))], fill=(235, 235, 240))
    d.ellipse((cx - 62, cy - 42, cx + 62, cy + 42), fill=(235, 235, 240))
    f = _font(56); bb = d.textbbox((0, 0), "400", font=f)
    d.text((cx - (bb[2] - bb[0]) / 2, cy - 34), "400", fill=(10, 14, 24), font=f)
    d.text((cx - 36, cy + 120), "km/h", fill=(160, 168, 180), font=_font(28))
    # right column: target distance bars, mode, level
    x0 = 650
    for k in range(12):
        h = 20 + 26 * (k % 4 == 0)
        d.rectangle((x0, 80 + k * 40, x0 + 30, 80 + k * 40 + 8), fill=(110, 120, 135))
    d.rectangle((x0 + 50, 110, x0 + 90, 560), outline=(110, 120, 135), width=2)
    d.rectangle((x0 + 52, 300, x0 + 88, 558), fill=(200, 200, 205))
    for k, (t, c) in enumerate((("CTCS-3", (120, 200, 255)), ("FS", (230, 230, 235)), ("GSM-R", (120, 220, 140)),
                                 ("07:12:48", (230, 230, 235)))):
        d.text((x0 + 120, 120 + k * 95), t, fill=c, font=_font(40))
    d.rectangle((0, H - 50, W, H), fill=(22, 28, 40))
    for k in range(8):
        d.rectangle((14 + k * 126, H - 44, 126 + k * 126, H - 8), outline=(80, 90, 105), width=2)
    p = os.path.join(texdir, "dmi_speed.png"); im.save(p); paths["speed"] = p
    # --- left: traction / brake / voltage
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    d.text((40, 30), "TRACTION / BRAKE", fill=(200, 205, 215), font=_font(36))
    for k, (lab, val, col) in enumerate((("TE", 0.62, (90, 200, 120)), ("BE", 0.0, (230, 90, 70)), ("U kV", 0.91, (120, 190, 255)),
                                          ("I A", 0.55, (240, 190, 70)))):
        y = 120 + k * 120
        d.text((40, y), lab, fill=(200, 205, 215), font=_font(34))
        d.rectangle((200, y, 960, y + 50), outline=(80, 90, 105), width=2)
        d.rectangle((202, y + 2, 202 + int(756 * val), y + 48), fill=col)
    d.text((40, H - 70), "27.5 kV   8 x 650 kW   MR 9.0 bar", fill=(160, 168, 180), font=_font(30))
    p = os.path.join(texdir, "dmi_traction.png"); im.save(p); paths["traction"] = p
    # --- right: line ahead (gradient profile, signals, neutral section)
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    d.text((40, 30), "LINE AHEAD", fill=(200, 205, 215), font=_font(36))
    d.line((80, 340, 980, 340), fill=(200, 200, 205), width=6)
    prof = [(80, 330), (300, 320), (480, 300), (640, 300), (820, 315), (980, 340)]
    d.line(prof, fill=(110, 180, 255), width=5)
    for xk in (240, 520, 800):
        d.rectangle((xk - 8, 360, xk + 8, 420), fill=(90, 210, 120))
    d.rectangle((600, 300, 700, 330), fill=(240, 190, 70))
    d.text((40, 470), "NEXT: TUNNEL 4.45 km   LIMIT 400", fill=(230, 230, 235), font=_font(34))
    d.text((40, 540), "BLOCK OK", fill=(90, 210, 120), font=_font(34))
    p = os.path.join(texdir, "dmi_map.png"); im.save(p); paths["map"] = p
    return paths


def _screen_mat(name, path):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(path, check_existing=True)
    uv = nt.nodes.new("ShaderNodeTexCoord"); L(uv.outputs["UV"], tx.inputs[0])
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 2.2
    L(tx.outputs["Color"], em.inputs["Color"])
    gl = nt.nodes.new("ShaderNodeBsdfPrincipled")              # cover glass: dark, glossy, reflects the windscreen
    gl.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1); gl.inputs["Roughness"].default_value = 0.08
    add = nt.nodes.new("ShaderNodeAddShader"); L(em.outputs[0], add.inputs[0]); L(gl.outputs[0], add.inputs[1])
    L(add.outputs[0], out.inputs[0])
    return m


def _console_mat():
    m = bpy.data.materials.get("CabConsole")
    if m:
        return m
    m = bpy.data.materials.new("CabConsole"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial"); p = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(p.outputs[0], out.inputs[0])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grain = nt.nodes.new("ShaderNodeTexNoise"); grain.inputs["Scale"].default_value = 900.0; grain.inputs["Detail"].default_value = 2.0
    L(tc.outputs["Object"], grain.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.15; bp.inputs["Distance"].default_value = 0.0005
    L(grain.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    dust = nt.nodes.new("ShaderNodeTexNoise"); dust.inputs["Scale"].default_value = 18.0; dust.inputs["Detail"].default_value = 6.0
    L(tc.outputs["Object"], dust.inputs["Vector"])
    dm = nt.nodes.new("ShaderNodeMapRange"); dm.inputs[1].default_value = 0.62; dm.inputs[2].default_value = 0.85
    L(dust.outputs["Fac"], dm.inputs[0])
    col = nt.nodes.new("ShaderNodeMix"); col.data_type = "RGBA"
    col.inputs["A"].default_value = (0.032, 0.034, 0.038, 1); col.inputs["B"].default_value = (0.055, 0.054, 0.052, 1)
    L(dm.outputs[0], col.inputs["Factor"]); L(col.outputs["Result"], p.inputs["Base Color"])
    rr = nt.nodes.new("ShaderNodeMapRange"); rr.inputs[3].default_value = 0.48; rr.inputs[4].default_value = 0.75
    L(dm.outputs[0], rr.inputs[0]); L(rr.outputs[0], p.inputs["Roughness"])
    return m


def build_cab(name, coll, X_origin, texdir):
    con = _console_mat()
    trim = M.principled("CabTrim", (0.06, 0.062, 0.068), 0.0, 0.4)
    bezel = M.principled("CabBezel", (0.012, 0.012, 0.014), 0.0, 0.35)
    seat = M.principled("CabSeat", (0.05, 0.05, 0.06), 0.0, 0.7)
    alu = M.principled("CabAlu", (0.55, 0.56, 0.58), 1.0, 0.3)
    btn = {c: M.principled(f"CabBtn_{c}", rgb, 0.0, 0.3, emission=rgb, estrength=e)
           for c, rgb, e in (("g", (0.15, 0.9, 0.3), 3.0), ("y", (1.0, 0.65, 0.1), 3.0), ("w", (0.9, 0.92, 1.0), 1.5),
                             ("r", (1.0, 0.12, 0.08), 0.0), ("k", (0.05, 0.05, 0.055), 0.0))}
    imgs = dmi_images(os.path.join(texdir, "cab"))
    scr = {k: _screen_mat(f"CabDMI_{k}", p) for k, p in imgs.items()}
    mats = [con, trim, bezel, seat, alu, *btn.values(), *scr.values()]
    mb = MB()
    X = lambda d: X_origin - d                                   # distance back from the nose tip
    # moulded desk: windscreen base (X-4.10, z 2.24) rising to the driver (X-5.55, z 2.52), full cab width
    prof = [(4.10, 2.24), (4.45, 2.31), (4.80, 2.38), (5.15, 2.46), (5.55, 2.52)]
    for (d0, z0), (d1, z1) in zip(prof[:-1], prof[1:]):
        mb.poly([(X(d0), -1.15, z0), (X(d0), 1.15, z0), (X(d1), 1.15, z1), (X(d1), -1.15, z1)], con)
    mb.poly([(X(5.55), -1.15, 2.52), (X(5.55), 1.15, 2.52), (X(5.55), 1.15, 1.6), (X(5.55), -1.15, 1.6)], con)
    # defroster slots along the windscreen base
    for k in range(14):
        y = -1.0 + k * 0.155
        mb.box(X(4.18), X(4.13), y, y + 0.11, 2.245, 2.262, bezel)
    # instrument hood with three displays tilted 35 deg towards the driver
    hood_d0, hood_d1 = 4.55, 5.05
    zb = 2.37
    mb.box(X(hood_d1), X(hood_d0), -0.95, 0.95, zb - 0.05, zb + 0.02, trim)
    tilt = math.radians(35.0)
    for (yc, key, w) in ((-0.62, "traction", 0.36), (0.0, "speed", 0.40), (0.62, "map", 0.36)):
        h = w * 0.625
        # screen plane: bottom edge near the driver, top edge leaning forward
        db = hood_d1 - 0.06; zb0 = zb + 0.03
        dt = db - h * math.sin(tilt); zt = zb0 + h * math.cos(tilt)
        pts = [(X(db), yc + w / 2, zb0), (X(db), yc - w / 2, zb0), (X(dt), yc - w / 2, zt), (X(dt), yc + w / 2, zt)]
        mb.poly(pts, scr[key])
        mb.uv = getattr(mb, "uv", {}); mb.uv[len(mb.f) - 1] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        # bezel frame (behind/around the glass) and the hood back
        e = 0.025; off = 0.004                       # bezel 25 mm wide, 4 mm behind the screen glass
        mb.poly([(X(db) + off, yc + w / 2 + e, zb0 - e), (X(db) + off, yc - w / 2 - e, zb0 - e),
                 (X(dt) + off, yc - w / 2 - e, zt + e), (X(dt) + off, yc + w / 2 + e, zt + e)], bezel)
        mb.poly([(X(dt), yc - w / 2 - e, zt + e), (X(dt), yc + w / 2 + e, zt + e),
                 (X(dt) + 0.12, yc + w / 2 + e, zt + e - 0.02), (X(dt) + 0.12, yc - w / 2 - e, zt + e - 0.02)], trim)
    # push-button strip and rotary switches on the far desk (in view against the windscreen base)
    cols = "gyywkkwgrkwkyg"
    for k, c in enumerate(cols):
        y = -0.85 + k * 0.13
        d0 = 4.33; z0 = 2.275
        mb.box(X(d0 + 0.035), X(d0), y, y + 0.045, z0, z0 + 0.018, btn[c] if c != "k" else btn["k"])
        mb.box(X(d0 + 0.05), X(d0 - 0.015), y - 0.01, y + 0.055, z0 - 0.004, z0 + 0.004, bezel)
    for y in (-0.98, 0.98):
        mb.cyl((X(4.40), y, 2.29), (X(4.40), y, 2.33), 0.03, alu, segs=16)
        mb.box(X(4.41), X(4.39), y - 0.028, y + 0.028, 2.33, 2.345, bezel)
    # traction / brake controller at the driver's left hand
    base = (X(5.42), -0.80, 2.50)
    mb.box(base[0] - 0.12, base[0] + 0.12, -0.90, -0.70, 2.48, 2.54, trim)
    mb.cyl(base, (base[0] + 0.10, base[1], base[2] + 0.16), 0.014, alu, segs=10)
    mb.cyl((base[0] + 0.10, base[1] - 0.07, base[2] + 0.16), (base[0] + 0.10, base[1] + 0.07, base[2] + 0.16), 0.022, btn["k"], segs=12)
    # seat back and the rear cab wall (out of shot, keeps the cab dark inside)
    mb.box(X(6.55), X(6.05), -0.28, 0.28, 1.4, 2.35, seat)
    mb.box(X(7.65), X(7.55), -1.6, 1.6, 1.2, 3.5, trim)
    idx = {m.name: i for i, m in enumerate(mats)}
    uvd = getattr(mb, "uv", {})
    mb.m = [idx[m.name] for m in mb.m]
    me = mb.make(name, mats, recalc=False)          # keep loop order: the display UVs depend on it
    uvl = me.uv_layers.new(name="UVMap")
    data = np.zeros((len(me.loops), 2), np.float32)
    for pi, uvs in uvd.items():
        p = me.polygons[pi]
        for k, li in enumerate(p.loop_indices):
            data[li] = uvs[k]
    uvl.data.foreach_set("uv", data.ravel())
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob
