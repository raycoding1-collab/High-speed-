"""Landscape: Guangxi-style tower karst valley at golden hour (bpy).

Terrain = natural ground (valley floor + hill massif at the tunnel end) modified by the railway
formation (cut / fill) along the alignment.  Karst towers are separate meshes.  Vegetation is
scattered with geometry nodes (Poly Haven CC0 models near cameras, procedural crowns mid-field).
"""
import math
import os

import bpy
import numpy as np

from . import alignment as A
from . import specs as S
from .bl import new_mesh_object, grid_faces

RNG_SEED = 7

# karst towers (x, y, height, base radius) - placed off the line, layered for depth
TOWERS = [
    # north side of the valley (behind the viaduct for the establishing shot)
    (300, 900, 160, 170), (820, 1250, 210, 190), (1350, 980, 140, 150), (1900, 1400, 240, 210), (2500, 1150, 175, 170),
    (3050, 1500, 220, 200), (2200, 700, 110, 120), (650, 1900, 260, 230), (1500, 2100, 300, 260), (2800, 2300, 280, 250),
    (3600, 1900, 250, 220), (-400, 1300, 190, 190), (4100, 1300, 200, 190), (-900, 2300, 280, 260), (3500, 2900, 320, 280),
    # south side (seen from the hill / cab / helicopter)
    (600, -800, 150, 160), (1400, -1100, 200, 190), (2300, -900, 170, 170), (3100, -1200, 230, 210), (200, -1600, 260, 240),
    (1900, -1900, 290, 260), (3800, -800, 180, 180), (-600, -900, 170, 170), (2800, -2200, 300, 270),
    # far ranges
    (5200, 2600, 380, 340), (6000, 900, 340, 320), (6200, -600, 300, 300), (4800, -2000, 360, 330), (-1800, 0, 300, 300),
    (-1500, 2500, 360, 330), (-1500, -2400, 340, 320), (1000, 3400, 400, 360), (2600, 3600, 420, 380), (600, -3000, 380, 350),
]


# ------------------------------------------------------------------ height fields
def _noise2(x, y, scale, seed):
    """Cheap smooth value noise (sum of sines with random phases) - deterministic, vectorised."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x, dtype=np.float64)
    amp = 1.0; f = 1.0 / scale; tot = 0
    for o in range(5):
        th = rng.uniform(0, 2 * np.pi, 3); ph = rng.uniform(0, 2 * np.pi, 3)
        for k in range(3):
            out += amp * np.sin(f * (x * np.cos(th[k]) + y * np.sin(th[k])) + ph[k]) / 3
        tot += amp; amp *= 0.5; f *= 2.03
    return out / tot


def _tower_r(i, ang, zf, R):
    """Radius of tower i at azimuth ang and relative height zf (0 base .. 1 top)."""
    return R * (1 + 0.20 * np.sin(3 * ang + i + 2.0 * zf) + 0.12 * np.sin(5 * ang + 2 * i - 3 * zf)
                + 0.07 * np.sin(9 * ang + 3 * i + 5 * zf) + 0.04 * np.sin(17 * ang + i + 9 * zf))


def tower_profile(i, d):
    """Height fraction vs normalised radius: sugar-loaf (steep flanks, rounded crown), varied per tower."""
    k = 1.8 + 0.8 * ((i * 37) % 7) / 6.0
    return np.clip(1 - d ** k, 0, 1) ** (0.55 + 0.25 * ((i * 13) % 5) / 4.0)


def tower_height(x, y):
    h = np.zeros_like(x, dtype=np.float64)
    for i, (tx, ty, H, R) in enumerate(TOWERS):
        dx = x - tx; dy = y - ty
        ang = np.arctan2(dy, dx)
        d = np.sqrt(dx * dx + dy * dy) / _tower_r(i, ang, 0.3, R)
        h = np.maximum(h, H * tower_profile(i, d))
    return h


RIVER_X = lambda t: 1650 + 120 * np.sin(t / 380) + 60 * np.sin(t / 150 + 1)


def river_depression(x, y):
    t = y + 500
    dx = x - RIVER_X(t)
    w = 34 + 6 * np.sin(t / 200)
    # channel + floodplain so that the coarse far grid also follows it
    return -2.4 * np.clip(1 - (np.abs(dx) / (w / 2 + 4)) ** 2, 0, 1) - 1.0 * np.clip(1 - np.abs(dx) / 90.0, 0, 1)


def natural_ground(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    valley = 1.5 * _noise2(x, y, 400, 1) + 0.6 * _noise2(x, y, 90, 2)
    # hill massif swallowing the tunnel (centred beyond the portal), with karst character
    s_t = A.TUNNEL_S
    tx, ty = A.xy(s_t + 380)
    d = np.sqrt((x - tx) ** 2 + ((y - ty) * 0.8) ** 2)
    massif = 165 * np.clip(1 - (d / 760) ** 2.0, 0, 1) ** 0.75 + 22 * _noise2(x, y, 60, 3) * np.clip(1 - d / 800, 0, 1)
    # foothills rising above rail level from ~s=3850 so the line runs in a cutting before the tunnel
    rise = np.clip((x - 3450) / 550, 0, 1) ** 1.3 * 44 * (1 + 0.25 * _noise2(x, y, 250, 4))
    return np.maximum(valley + rise + river_depression(x, y), massif)


def formation(s, lat, g):
    """Apply railway earthworks: fill (embankment) and cut (cutting) around the line where not on viaduct."""
    form_z = A.RAIL_Z - 0.85                 # formation level below rail
    half = 7.0                               # formation half width (double track + cess + ditch)
    out = g.copy()
    on = (s > A.VIADUCT[1] - 20) & (s < A.TUNNEL_S + 15)
    al = np.abs(lat)
    # cutting slope 1:1 (soil) / embankment slope 1:1.5
    ex = np.maximum(al - half, 0)
    tier = np.floor(ex / 10.0)                       # 1:1 slope, 2 m bench every 8 m of height
    cut = form_z + tier * 8.0 + np.minimum(ex - tier * 10.0, 8.0)
    fill = form_z - np.maximum(al - half, 0) / 1.5
    target = np.where(g > form_z, np.minimum(g, cut), np.maximum(g, fill))
    # ditches at the cutting toe
    ditch = (al > half - 1.2) & (al < half - 0.3) & (g > form_z)
    target = np.where(ditch, target - 0.5, target)
    w = np.clip((s - (A.VIADUCT[1] - 20)) / 30, 0, 1) * np.clip((A.TUNNEL_S + 15 - s) / 10, 0, 1)
    out = np.where(on, g * (1 - w) + target * w, g)
    return out


def ground(x, y):
    g = natural_ground(x, y)
    s, lat = A.nearest_s(np.ravel(x), np.ravel(y))
    s = s.reshape(np.shape(x)); lat = lat.reshape(np.shape(x))
    return formation(s, lat, g)


# ------------------------------------------------------------------ meshes
def build_terrain(coll, mats):
    """Far terrain grid (hole along the corridor) + dense corridor strip."""
    # far grid
    xs = np.arange(-4000, 8000.1, 30.0); ys = np.arange(-4500, 5000.1, 30.0)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = natural_ground(X, Y)
    s, lat = A.nearest_s(X.ravel(), Y.ravel())
    s = s.reshape(X.shape); lat = lat.reshape(X.shape)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    ny, nx = X.shape
    faces = []
    inside = (np.abs(lat) < 150) & (s > A.S_MIN + 10) & (s < A.S_MAX - 10)
    for j in range(ny - 1):
        for i in range(nx - 1):
            if inside[j, i] and inside[j + 1, i + 1] and inside[j, i + 1] and inside[j + 1, i]:
                continue
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    far = new_mesh_object("TerrainFar", V, faces, coll, smooth=True)
    far.data.materials.append(mats["terrain"])
    # corridor strip
    lats = np.array(sorted(set(np.round(np.concatenate([
        -np.geomspace(6.0, 160, 34), np.geomspace(6.0, 160, 34), np.linspace(-5.5, 5.5, 8)]), 3))))
    ss = np.arange(A.S_MIN + 5, A.S_MAX - 5, 2.0)
    th = A.heading(ss); cx, cy = A.xy(ss)
    PX = cx[:, None] - np.sin(th)[:, None] * lats[None, :]
    PY = cy[:, None] + np.cos(th)[:, None] * lats[None, :]
    G = natural_ground(PX, PY)
    G = formation(np.repeat(ss[:, None], len(lats), 1), np.repeat(lats[None, :], len(ss), 0), G)
    V = np.stack([PX, PY, G], -1).reshape(-1, 3)
    n = len(lats)
    faces = []
    for i in range(len(ss) - 1):
        for j in range(n - 1):
            # leave the tunnel bore open beyond the portal (hill surface stays, ground under track removed)
            a = i * n + j
            faces.append((a, a + n, a + n + 1, a + 1))       # wound so normals point up
    nlat = len(lats)
    uv = []
    for (a, b, c, d) in faces:
        for vi in (a, b, c, d):
            i, j = divmod(vi, nlat)
            uv.append((ss[i], lats[j]))
    strip = new_mesh_object("TerrainCorridor", V, faces, coll, smooth=True, uvs=np.array(uv))
    nat = natural_ground(PX, PY).ravel()
    S2d = np.repeat(ss[:, None], n, 1).ravel(); L2d = np.repeat(lats[None, :], len(ss), 0).ravel()
    cutm = ((nat > V[:, 2] + 0.3) & (np.abs(L2d) > 7.2) & (S2d > A.VIADUCT[1]) & (S2d < A.TUNNEL_S + 5)).astype(np.float32)
    at = strip.data.attributes.new("cutmask", "FLOAT", "POINT"); at.data.foreach_set("value", cutm)
    strip.data.materials.append(mats["terrain"])
    return far, strip


def build_towers(coll, mats):
    """Each tower: polar-grid sugar-loaf with height-varying lobes, ledges and overhang-free noise."""
    objs = []
    for i, (tx, ty, H, R) in enumerate(TOWERS):
        nr, na = 40, 128
        r = np.linspace(0, 1.3, nr) ** 0.9
        a = np.linspace(0, 2 * np.pi, na, endpoint=False)
        RR, AA = np.meshgrid(r, a, indexing="ij")
        prof = tower_profile(i, RR)
        zf = prof
        rr = _tower_r(i, AA, zf, R)
        X = tx + RR * rr * np.cos(AA); Y = ty + RR * rr * np.sin(AA)
        ledges = 0.025 * H * np.sin(RR * 31 + AA * 3 + i) * (RR < 1) * (prof > 0.15)
        bumps = 0.05 * H * _noise2(X, Y, 45, 100 + i) * (RR < 1.05)
        g = natural_ground(X, Y)
        Z = H * prof + ledges + bumps * prof + g - 3.0
        Z = np.maximum(Z, g - 3.0)
        V = np.stack([X, Y, Z], -1).reshape(-1, 3)
        faces = []
        for p in range(nr - 1):
            for q in range(na):
                q2 = (q + 1) % na
                faces.append((p * na + q, p * na + q2, (p + 1) * na + q2, (p + 1) * na + q))
        ob = new_mesh_object(f"Tower{i:02d}", V, faces, coll, smooth=True)
        ob.data.materials.append(mats["karst"])
        objs.append(ob)
    return objs


# ------------------------------------------------------------------ materials
def _tex(nt, path, noncolor=False):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(path, check_existing=True)
    if noncolor:
        n.image.colorspace_settings.name = "Non-Color"
    return n


def terrain_material(tex, paddy_img):
    """Valley paddies (from layout map), grassy slopes, forest floor, rock on steep faces."""
    m = bpy.data.materials.new("Terrain")
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Position"], sep.inputs[0])
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], nsep.inputs[0])
    # world-space tiling for detail textures
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (0.35, 0.35, 0.35)
    L(geo.outputs["Position"], mp.inputs[0])
    grass0 = _tex(nt, tex["grass"]["diff"]); L(mp.outputs[0], grass0.inputs[0])
    gbw = nt.nodes.new("ShaderNodeRGBToBW"); L(grass0.outputs["Color"], gbw.inputs[0])
    grass = nt.nodes.new("ShaderNodeMix"); grass.data_type = "RGBA"; grass.blend_type = "MULTIPLY"; grass.inputs["Factor"].default_value = 1.0
    grass.inputs["B"].default_value = (0.36, 0.62, 0.15, 1)
    L(gbw.outputs[0], grass.inputs["A"])
    forest = _tex(nt, tex["forest"]["diff"]); L(mp.outputs[0], forest.inputs[0])
    rock0 = _tex(nt, tex["cliff"]["diff"]); L(mp.outputs[0], rock0.inputs[0])
    rock = nt.nodes.new("ShaderNodeHueSaturation"); rock.inputs["Saturation"].default_value = 0.3; rock.inputs["Value"].default_value = 1.2
    L(rock0.outputs["Color"], rock.inputs["Color"])
    mud = _tex(nt, tex["mud"]["diff"]); L(mp.outputs[0], mud.inputs[0])
    nor = _tex(nt, tex["grass"]["nor"], True); L(mp.outputs[0], nor.inputs[0])
    # paddy map in world XY over the valley
    pm = nt.nodes.new("ShaderNodeMapping"); pm.vector_type = "POINT"
    pm.inputs["Location"].default_value = (-PADDY_X0 / PADDY_W, -PADDY_Y0 / PADDY_H, 0)
    pm.inputs["Scale"].default_value = (1 / PADDY_W, 1 / PADDY_H, 1)
    L(geo.outputs["Position"], pm.inputs[0])
    # Mapping applies scale before location: so (P*scale + loc) -> use separate math instead
    pm.inputs["Location"].default_value = (-PADDY_X0 / PADDY_W, -PADDY_Y0 / PADDY_H, 0)
    pimg = nt.nodes.new("ShaderNodeTexImage"); pimg.image = paddy_img; pimg.interpolation = "Linear"; pimg.extension = "CLIP"
    pimg.image.colorspace_settings.name = "Non-Color"
    L(pm.outputs[0], pimg.inputs[0])
    psep = nt.nodes.new("ShaderNodeSeparateColor"); L(pimg.outputs["Color"], psep.inputs[0])
    field_var, dike, ftype = psep.outputs[0], psep.outputs[1], psep.outputs[2]
    def cmp(v):
        c = nt.nodes.new("ShaderNodeMath"); c.operation = "COMPARE"; c.inputs[1].default_value = v; c.inputs[2].default_value = 0.1
        L(ftype, c.inputs[0]); return c.outputs[0]
    is_ripe, water, is_fallow = cmp(0.33), cmp(0.66), cmp(1.0)
    # slope factor
    slope = nt.nodes.new("ShaderNodeMapRange"); slope.inputs[1].default_value = 0.30; slope.inputs[2].default_value = 0.45
    slope.inputs[3].default_value = 1.0; slope.inputs[4].default_value = 0.0
    L(nsep.outputs["Z"], slope.inputs[0])
    # elevation-based: valley (z<6) -> paddy zone
    valley = nt.nodes.new("ShaderNodeMapRange"); valley.inputs[1].default_value = 4.0; valley.inputs[2].default_value = 9.0
    valley.inputs[3].default_value = 1.0; valley.inputs[4].default_value = 0.0
    L(sep.outputs["Z"], valley.inputs[0])
    flat = nt.nodes.new("ShaderNodeMapRange"); flat.inputs[1].default_value = 0.96; flat.inputs[2].default_value = 0.99
    L(nsep.outputs["Z"], flat.inputs[0])
    pz = nt.nodes.new("ShaderNodeMath"); pz.operation = "MULTIPLY"; L(valley.outputs[0], pz.inputs[0]); L(flat.outputs[0], pz.inputs[1])
    pz2 = nt.nodes.new("ShaderNodeMath"); pz2.operation = "MULTIPLY"; L(pz.outputs[0], pz2.inputs[0]); L(pimg.outputs["Alpha"], pz2.inputs[1])
    # rice colour: young green to yellow-green by field variation
    rice = nt.nodes.new("ShaderNodeValToRGB")
    rice.color_ramp.elements[0].color = (0.075, 0.15, 0.028, 1); rice.color_ramp.elements[1].color = (0.11, 0.19, 0.035, 1)
    rice.color_ramp.elements.new(0.55).color = (0.09, 0.17, 0.03, 1)
    L(field_var, rice.inputs[0])
    ripe = nt.nodes.new("ShaderNodeMix"); ripe.data_type = "RGBA"; ripe.inputs["B"].default_value = (0.17, 0.20, 0.05, 1)
    L(is_ripe, ripe.inputs["Factor"]); L(rice.outputs[0], ripe.inputs["A"])
    fallow = nt.nodes.new("ShaderNodeMix"); fallow.data_type = "RGBA"
    L(is_fallow, fallow.inputs["Factor"]); L(ripe.outputs["Result"], fallow.inputs["A"]); L(mud.outputs["Color"], fallow.inputs["B"])
    riceN = nt.nodes.new("ShaderNodeTexNoise"); riceN.inputs["Scale"].default_value = 3.0; riceN.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], riceN.inputs["Vector"])
    ricem = nt.nodes.new("ShaderNodeMix"); ricem.data_type = "RGBA"; ricem.blend_type = "MULTIPLY"; ricem.inputs["Factor"].default_value = 0.6
    L(fallow.outputs["Result"], ricem.inputs["A"]); L(riceN.outputs["Color"], ricem.inputs["B"])
    # hillside: grass <-> forest floor (dark canopy from afar), rock on steep
    hill = nt.nodes.new("ShaderNodeMix"); hill.data_type = "RGBA"
    big = nt.nodes.new("ShaderNodeTexNoise"); big.inputs["Scale"].default_value = 0.01; big.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], big.inputs["Vector"])
    L(big.outputs["Fac"], hill.inputs["Factor"]); L(grass.outputs["Result"], hill.inputs["A"])
    canopy = nt.nodes.new("ShaderNodeMix"); canopy.data_type = "RGBA"; canopy.blend_type = "MULTIPLY"; canopy.inputs["Factor"].default_value = 1.0
    canopy.inputs["B"].default_value = (0.30, 0.55, 0.16, 1)
    L(forest.outputs["Color"], canopy.inputs["A"]); L(canopy.outputs["Result"], hill.inputs["B"])
    rk = nt.nodes.new("ShaderNodeMix"); rk.data_type = "RGBA"
    L(slope.outputs[0], rk.inputs["Factor"]); L(hill.outputs["Result"], rk.inputs["A"]); L(rock.outputs["Color"], rk.inputs["B"])
    # paddy: rice vs water vs dikes
    pad = nt.nodes.new("ShaderNodeMix"); pad.data_type = "RGBA"
    dk = nt.nodes.new("ShaderNodeMath"); dk.operation = "POWER"; dk.inputs[1].default_value = 0.6; L(dike, dk.inputs[0])
    L(dk.outputs[0], pad.inputs["Factor"]); L(ricem.outputs["Result"], pad.inputs["A"]); L(grass.outputs["Result"], pad.inputs["B"])
    allc = nt.nodes.new("ShaderNodeMix"); allc.data_type = "RGBA"
    L(pz2.outputs[0], allc.inputs["Factor"]); L(rk.outputs["Result"], allc.inputs["A"]); L(pad.outputs["Result"], allc.inputs["B"])
    # diamond lattice concrete frames (slope protection) on cut slopes: uses UV = (chainage, lateral)
    uvn = nt.nodes.new("ShaderNodeUVMap")
    usep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(uvn.outputs[0], usep.inputs[0])
    def diag(sign):
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY_ADD"; a.inputs[1].default_value = sign
        L(usep.outputs["Y"], a.inputs[0]); L(usep.outputs["X"], a.inputs[2])
        b = nt.nodes.new("ShaderNodeMath"); b.operation = "PINGPONG"; b.inputs[1].default_value = 2.0; L(a.outputs[0], b.inputs[0])
        c = nt.nodes.new("ShaderNodeMath"); c.operation = "LESS_THAN"; c.inputs[1].default_value = 0.28; L(b.outputs[0], c.inputs[0])
        return c.outputs[0]
    lat1, lat2 = diag(1.0), diag(-1.0)
    latt = nt.nodes.new("ShaderNodeMath"); latt.operation = "MAXIMUM"; L(lat1, latt.inputs[0]); L(lat2, latt.inputs[1])
    cm = nt.nodes.new("ShaderNodeAttribute"); cm.attribute_name = "cutmask"
    lm = nt.nodes.new("ShaderNodeMath"); lm.operation = "MULTIPLY"; L(latt.outputs[0], lm.inputs[0]); L(cm.outputs["Fac"], lm.inputs[1])
    cutgrass = nt.nodes.new("ShaderNodeMix"); cutgrass.data_type = "RGBA"; L(cm.outputs["Fac"], cutgrass.inputs["Factor"])
    L(allc.outputs["Result"], cutgrass.inputs["A"]); L(grass.outputs["Result"], cutgrass.inputs["B"])
    frame_col = nt.nodes.new("ShaderNodeMix"); frame_col.data_type = "RGBA"; frame_col.inputs["B"].default_value = (0.26, 0.255, 0.24, 1)
    L(lm.outputs[0], frame_col.inputs["Factor"]); L(cutgrass.outputs["Result"], frame_col.inputs["A"])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(frame_col.outputs["Result"], p.inputs["Base Color"]); p.inputs["Roughness"].default_value = 0.85
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
    L(nor.outputs["Color"], nm.inputs["Color"]); L(nm.outputs[0], p.inputs["Normal"])
    # water in flooded paddies: mirror-like with slight ripples, muddy tint
    wat = nt.nodes.new("ShaderNodeBsdfPrincipled")
    wat.inputs["Base Color"].default_value = (0.05, 0.045, 0.03, 1); wat.inputs["Roughness"].default_value = 0.18
    wat.inputs["IOR"].default_value = 1.33
    wat.inputs["Specular IOR Level"].default_value = 0.42
    rip = nt.nodes.new("ShaderNodeTexNoise"); rip.inputs["Scale"].default_value = 1.2; rip.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], rip.inputs["Vector"])
    bmp = nt.nodes.new("ShaderNodeBump"); bmp.inputs["Strength"].default_value = 0.04; L(rip.outputs["Fac"], bmp.inputs["Height"])
    L(bmp.outputs[0], wat.inputs["Normal"])
    wmask = nt.nodes.new("ShaderNodeMath"); wmask.operation = "MULTIPLY"
    nd = nt.nodes.new("ShaderNodeMath"); nd.operation = "SUBTRACT"; nd.inputs[0].default_value = 1.0; L(dike, nd.inputs[1])
    wm2 = nt.nodes.new("ShaderNodeMath"); wm2.operation = "MULTIPLY"; L(water, wm2.inputs[0]); L(nd.outputs[0], wm2.inputs[1])
    L(wm2.outputs[0], wmask.inputs[0]); L(pz2.outputs[0], wmask.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader"); L(wmask.outputs[0], mix.inputs[0]); L(p.outputs[0], mix.inputs[1]); L(wat.outputs[0], mix.inputs[2])
    L(mix.outputs[0], out.inputs[0])
    return m


def karst_material(tex):
    """Tower karst: dense subtropical canopy with grey limestone cliffs (water-stained) on the steepest faces."""
    m = bpy.data.materials.new("Karst")
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], nsep.inputs[0])
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (0.05, 0.05, 0.05); L(geo.outputs["Position"], mp.inputs[0])
    rock = _tex(nt, tex["cliff"]["diff"]); L(mp.outputs[0], rock.inputs[0])
    grey = nt.nodes.new("ShaderNodeHueSaturation"); grey.inputs["Saturation"].default_value = 0.25; grey.inputs["Value"].default_value = 1.25
    L(rock.outputs["Color"], grey.inputs["Color"])
    st = nt.nodes.new("ShaderNodeTexNoise"); st.inputs["Scale"].default_value = 0.15; st.inputs["Detail"].default_value = 2
    smap = nt.nodes.new("ShaderNodeMapping"); smap.inputs["Scale"].default_value = (1.0, 1.0, 0.06)
    L(geo.outputs["Position"], smap.inputs[0]); L(smap.outputs[0], st.inputs["Vector"])
    stk = nt.nodes.new("ShaderNodeValToRGB"); stk.color_ramp.elements[0].color = (0.25, 0.24, 0.22, 1); stk.color_ramp.elements[1].color = (1.0, 0.98, 0.95, 1)
    L(st.outputs["Fac"], stk.inputs[0])
    rcol = nt.nodes.new("ShaderNodeMix"); rcol.data_type = "RGBA"; rcol.blend_type = "MULTIPLY"; rcol.inputs["Factor"].default_value = 1.0
    L(grey.outputs["Color"], rcol.inputs["A"]); L(stk.outputs[0], rcol.inputs["B"])
    # rock exposure: steep (normal z small) AND patchy noise
    pn = nt.nodes.new("ShaderNodeTexNoise"); pn.inputs["Scale"].default_value = 0.012; pn.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], pn.inputs["Vector"])
    steep = nt.nodes.new("ShaderNodeMath"); steep.operation = "MULTIPLY_ADD"; steep.inputs[1].default_value = -1.2; steep.inputs[2].default_value = 1.05
    L(nsep.outputs["Z"], steep.inputs[0])                 # ~1 on vertical faces, <0 on gentle slopes
    ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "MULTIPLY_ADD"; ex.inputs[1].default_value = 1.4
    L(pn.outputs["Fac"], ex.inputs[0]); L(steep.outputs[0], ex.inputs[2])
    expo = nt.nodes.new("ShaderNodeMapRange"); expo.inputs[1].default_value = 1.55; expo.inputs[2].default_value = 1.75
    L(ex.outputs[0], expo.inputs[0])
    # canopy: crowns as voronoi clumps, colour variation
    canopyN = nt.nodes.new("ShaderNodeTexVoronoi"); canopyN.inputs["Scale"].default_value = 0.18
    L(geo.outputs["Position"], canopyN.inputs["Vector"])
    var = nt.nodes.new("ShaderNodeTexNoise"); var.inputs["Scale"].default_value = 0.02; var.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], var.inputs["Vector"])
    can = nt.nodes.new("ShaderNodeValToRGB")
    can.color_ramp.elements[0].color = (0.020, 0.045, 0.012, 1); can.color_ramp.elements[1].color = (0.085, 0.14, 0.03, 1)
    can.color_ramp.elements.new(0.5).color = (0.05, 0.10, 0.02, 1)
    mixv = nt.nodes.new("ShaderNodeMath"); mixv.operation = "MULTIPLY_ADD"; mixv.inputs[1].default_value = 0.6
    L(canopyN.outputs["Distance"], mixv.inputs[0]); L(var.outputs["Fac"], mixv.inputs[2])
    L(mixv.outputs[0], can.inputs[0])
    col = nt.nodes.new("ShaderNodeMix"); col.data_type = "RGBA"
    L(expo.outputs[0], col.inputs["Factor"]); L(can.outputs[0], col.inputs["A"]); L(rcol.outputs["Result"], col.inputs["B"])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(col.outputs["Result"], p.inputs["Base Color"])
    rg = nt.nodes.new("ShaderNodeMapRange"); rg.inputs[3].default_value = 0.9; rg.inputs[4].default_value = 0.75; L(expo.outputs[0], rg.inputs[0])
    L(rg.outputs[0], p.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 1.0; bump.inputs["Distance"].default_value = 2.5
    L(canopyN.outputs["Distance"], bump.inputs["Height"]); L(bump.outputs[0], p.inputs["Normal"])
    L(p.outputs[0], out.inputs[0])
    return m


# ------------------------------------------------------------------ paddy layout
PADDY_X0, PADDY_Y0, PADDY_W, PADDY_H = -1500.0, -1500.0, 6000.0, 3500.0


def paddy_image(path, res=1.0, seed=3):
    """Field layout map from jittered Voronoi cells (fields ~15-40 m), stretched along the valley.
    R = per-field variation, G = dike mask, B = field type (0 rice, .33 ripening, .66 flooded, 1 fallow), A = 1."""
    from PIL import Image, ImageFilter
    from scipy.spatial import cKDTree
    W = int(PADDY_W / res); H = int(PADDY_H / res)
    rng = np.random.default_rng(seed)
    n = int(PADDY_W * PADDY_H / (26.0 * 18.0))
    seeds = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])
    stretch = np.array([0.62, 1.0])                       # fields elongated along x
    tree = cKDTree(seeds * stretch)
    ft = rng.choice([0.0, 0.33, 0.66, 1.0], n, p=[0.70, 0.22, 0.05, 0.03])
    var = rng.uniform(0, 1, n)
    R = np.zeros((H, W), np.uint8); G = np.zeros((H, W), np.uint8); B = np.zeros((H, W), np.uint8)
    for y0 in range(0, H, 500):
        ys = np.arange(y0, min(H, y0 + 500))
        xx, yy = np.meshgrid(np.arange(W) + 0.5, ys + 0.5)
        # warp coordinates slightly so boundaries are not straight
        wx = xx + 6 * np.sin(yy / 23.0) + 3 * np.sin(xx / 11.0 + yy / 17.0)
        wy = yy + 5 * np.sin(xx / 29.0) + 2 * np.sin(yy / 7.0)
        d, idx = tree.query(np.column_stack([wx.ravel() * stretch[0], wy.ravel() * stretch[1]]), k=2)
        edge = (d[:, 1] - d[:, 0]) < 0.9
        idx0 = idx[:, 0]
        R[ys] = (var[idx0] * 255).reshape(len(ys), W)
        B[ys] = (ft[idx0] * 255).reshape(len(ys), W)
        G[ys] = (edge * 255).reshape(len(ys), W)
    g = Image.fromarray(G).filter(ImageFilter.GaussianBlur(0.7))
    img = Image.merge("RGBA", (Image.fromarray(R), g, Image.fromarray(B), Image.fromarray(np.full((H, W), 255, np.uint8))))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)
    return path


# ------------------------------------------------------------------ river
def build_river(coll):
    """Meandering river crossing under the viaduct (north-south), water surface mesh."""
    t = np.linspace(-1500, 2500, 400)
    xr = RIVER_X(t)
    yr = t - 500
    width = 34 + 6 * np.sin(t / 200)
    dx = np.gradient(xr); dy = np.gradient(yr); n = np.hypot(dx, dy)
    nx, ny = -dy / n, dx / n
    V = []
    for k in range(len(t)):
        for f in (-1, 1):
            px, py = xr[k] + nx[k] * f * width[k] / 2, yr[k] + ny[k] * f * width[k] / 2
            V.append((px, py, -1.9))
    faces = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(len(t) - 1)]
    faces = [(a, d, c, b) for (a, b, c, d) in faces]      # counter-clockwise from above -> normals up
    ob = new_mesh_object("River", np.array(V), faces, coll, smooth=True)
    if np.mean([p.normal.z for p in ob.data.polygons]) < 0:
        ob.data.flip_normals()
    m = bpy.data.materials.new("RiverWater")
    m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.03, 0.045, 0.035, 1); p.inputs["Roughness"].default_value = 0.04
    p.inputs["IOR"].default_value = 1.33
    geo = nt.nodes.new("ShaderNodeNewGeometry"); wn = nt.nodes.new("ShaderNodeTexWave"); wn.inputs["Scale"].default_value = 0.08
    wn.inputs["Distortion"].default_value = 6
    L = nt.links.new; L(geo.outputs["Position"], wn.inputs["Vector"])
    bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.05
    L(wn.outputs["Fac"], bm.inputs["Height"]); L(bm.outputs[0], p.inputs["Normal"])
    ob.data.materials.append(m)
    return ob, (xr, yr, width)


# ------------------------------------------------------------------ vegetation
def load_models(assets, names):
    """Append objects from the Poly Haven .blend files into a hidden source collection."""
    from .bl import collection
    src = collection("VegSources")
    src.hide_render = False
    out = {}
    for key, (asset, objname, target_h) in names.items():
        f = os.path.join(assets, "models", asset, f"{asset}_1k.blend")
        with bpy.data.libraries.load(f, link=False) as (dfrom, dto):
            dto.objects = [n for n in dfrom.objects if n == objname]
        for o in dto.objects:
            if o is None:
                continue
            src.objects.link(o)
            o.location = (0, 0, -10000)      # park far away (instancing uses ORIGINAL space: relative)
            o.location = (0, 0, 0)
            h = max(o.dimensions.z, 1e-3)
            o.scale = tuple(v * target_h / h for v in o.scale)
            out[key] = o
    for o in src.objects:
        o.hide_render = True
    src.hide_viewport = True
    return out, src


def scatter_tree(name, coll, pts_xyz, sources, scale_range, seed, wind=0.03):
    """Instance a random pick of `sources` objects on points with random yaw/scale + wind sway (GN)."""
    from .bl import collection
    rng = np.random.default_rng(seed)
    n = len(pts_xyz)
    me = bpy.data.meshes.new(name + "_pts")
    me.vertices.add(n); me.vertices.foreach_set("co", np.asarray(pts_xyz, np.float32).ravel())
    pick = rng.integers(0, len(sources), n)
    a = me.attributes.new("pick", "INT", "POINT"); a.data.foreach_set("value", pick.astype(np.int32))
    sc = rng.uniform(*scale_range, n).astype(np.float32)
    b = me.attributes.new("sc", "FLOAT", "POINT"); b.data.foreach_set("value", sc)
    yaw = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    c = me.attributes.new("yaw", "FLOAT", "POINT"); c.data.foreach_set("value", yaw)
    me.update()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    vc = collection(name + "_src", bpy.data.collections["VegSources"])
    for o in sources:
        if o.name not in vc.objects:
            vc.objects.link(o)
    md = ob.modifiers.new("scatter", "NODES"); md.node_group = _scatter_tree()
    from .bl import _set_modifier_input
    _set_modifier_input(md, "Collection", vc)
    _set_modifier_input(md, "Wind", wind)
    return ob


def _scatter_tree():
    ng = bpy.data.node_groups.get("VegScatter")
    if ng:
        return ng
    ng = bpy.data.node_groups.new("VegScatter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Collection", in_out="INPUT", socket_type="NodeSocketCollection")
    w = ng.interface.new_socket("Wind", in_out="INPUT", socket_type="NodeSocketFloat"); w.default_value = 0.03
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; L = ng.links.new
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeCollectionInfo"); ci.transform_space = "ORIGINAL"
    ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
    L(gi.outputs["Collection"], ci.inputs["Collection"])
    pick = N.new("GeometryNodeInputNamedAttribute"); pick.data_type = "INT"; pick.inputs["Name"].default_value = "pick"
    scl = N.new("GeometryNodeInputNamedAttribute"); scl.data_type = "FLOAT"; scl.inputs["Name"].default_value = "sc"
    yaw = N.new("GeometryNodeInputNamedAttribute"); yaw.data_type = "FLOAT"; yaw.inputs["Name"].default_value = "yaw"
    # wind: tilt angles from 4D noise (position, time)
    pos = N.new("GeometryNodeInputPosition"); tm = N.new("GeometryNodeInputSceneTime")
    nz = N.new("ShaderNodeTexNoise"); nz.noise_dimensions = "4D"; nz.inputs["Scale"].default_value = 0.08; nz.inputs["Detail"].default_value = 1.0
    tmul = N.new("ShaderNodeMath"); tmul.operation = "MULTIPLY"; tmul.inputs[1].default_value = 0.55
    L(tm.outputs["Seconds"], tmul.inputs[0]); L(pos.outputs[0], nz.inputs["Vector"]); L(tmul.outputs[0], nz.inputs["W"])
    sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    L(nz.outputs["Color"], sub.inputs[0])
    wmul = N.new("ShaderNodeVectorMath"); wmul.operation = "SCALE"
    L(sub.outputs[0], wmul.inputs[0]); L(gi.outputs["Wind"], wmul.inputs["Scale"])
    sepw = N.new("ShaderNodeSeparateXYZ"); L(wmul.outputs[0], sepw.inputs[0])
    comb = N.new("ShaderNodeCombineXYZ"); L(sepw.outputs["X"], comb.inputs["X"]); L(sepw.outputs["Y"], comb.inputs["Y"]); L(yaw.outputs["Attribute"], comb.inputs["Z"])
    e2r = N.new("FunctionNodeEulerToRotation"); L(comb.outputs[0], e2r.inputs["Euler"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True
    L(gi.outputs["Geometry"], iop.inputs["Points"]); L(ci.outputs[0], iop.inputs["Instance"])
    L(pick.outputs["Attribute"], iop.inputs["Instance Index"]); L(e2r.outputs[0], iop.inputs["Rotation"])
    L(scl.outputs["Attribute"], iop.inputs["Scale"])
    L(iop.outputs[0], go.inputs["Geometry"])
    return ng


def crown_proxy(name, coll, seed=0):
    """Mid-distance tree proxy: displaced crown + trunk, leafy shader (no alpha cards)."""
    import bmesh
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    for v in bm.verts:
        p = np.array(v.co)
        n = 1 + 0.22 * math.sin(3 * p[0] + seed) * math.cos(2.5 * p[1]) + 0.15 * math.sin(4 * p[2] + seed * 2) + rng.uniform(-0.06, 0.06)
        v.co = (p[0] * n * 1.0, p[1] * n * 1.0, p[2] * n * 0.8 + 1.6)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    m = bpy.data.materials.get("CrownLeaves")
    if m is None:
        m = bpy.data.materials.new("CrownLeaves"); m.use_nodes = True
        nt = m.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
        vo = nt.nodes.new("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = 9.0
        cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.elements[0].color = (0.02, 0.05, 0.012, 1); cr.color_ramp.elements[1].color = (0.11, 0.19, 0.04, 1)
        L(vo.outputs["Distance"], cr.inputs[0]); L(cr.outputs[0], p.inputs["Base Color"])
        bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 1.0; bp.inputs["Distance"].default_value = 0.25
        L(vo.outputs["Distance"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
        p.inputs["Roughness"].default_value = 0.75
        p.inputs["Subsurface Weight"].default_value = 0.15
        p.inputs["Subsurface Radius"].default_value = (0.3, 0.6, 0.1)
    ob.data.materials.append(m)
    trunk = bpy.data.meshes.new(name + "_t")
    return ob


def sample_surface(obj_list, density, rng, filt=None, max_n=200000):
    """Random points on the evaluated surfaces of objects (area weighted), optional filter f(p, n)."""
    pts_all = []
    for ob in obj_list:
        me = ob.data
        co = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        me.calc_loop_triangles()
        tri = np.zeros(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
        M = np.array(ob.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
        a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
        nrm = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(nrm, axis=1); nrm /= (2 * area[:, None] + 1e-9)
        n = min(max_n, int(area.sum() * density))
        if n <= 0:
            continue
        idx = rng.choice(len(tri), n, p=area / area.sum())
        r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
        p = a[idx] * (1 - r1)[:, None] + b[idx] * (r1 * (1 - r2))[:, None] + c[idx] * (r1 * r2)[:, None]
        nn = nrm[idx]
        if filt is not None:
            keep = filt(p, nn)
            p = p[keep]
        pts_all.append(p)
    return np.concatenate(pts_all) if pts_all else np.zeros((0, 3))


# ------------------------------------------------------------------ sky / sun / haze
def build_sky(scene, hdri_path, sun_azimuth_deg=262.0, exposure=0.0, strength=0.9):
    """HDRI sky rotated so its (clipped) sun lies at the requested azimuth, plus a matched warm sun lamp."""
    HDRI_SUN_U, HDRI_SUN_ELEV = 0.612, 4.64
    phi0 = (HDRI_SUN_U - 0.5) * 2 * math.pi
    d0 = np.array([-math.cos(phi0), math.sin(phi0)])            # horizontal dir of the HDRI sun (blender equirect)
    az0 = math.degrees(math.atan2(d0[0], d0[1])) % 360          # azimuth from north (clockwise)
    rot = math.radians(az0 - sun_azimuth_deg)                   # rotate environment by -(...) via mapping
    w = bpy.data.worlds.new("World")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear(); L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Rotation"].default_value = (0, 0, -rot)
    L(tc.outputs["Generated"], mp.inputs[0])
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdri_path, check_existing=True)
    L(mp.outputs[0], env.inputs[0])
    # clamp the clipped sun disc in the HDRI (the lamp provides the direct light)
    sep = nt.nodes.new("ShaderNodeSeparateColor"); L(env.outputs[0], sep.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineColor")
    for i in range(3):
        mn = nt.nodes.new("ShaderNodeMath"); mn.operation = "MINIMUM"; mn.inputs[1].default_value = 6.0
        L(sep.outputs[i], mn.inputs[0]); L(mn.outputs[0], comb.inputs[i])
    bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = strength
    L(comb.outputs[0], bg.inputs[0])
    out = nt.nodes.new("ShaderNodeOutputWorld"); L(bg.outputs[0], out.inputs[0])
    # sun lamp
    el = math.radians(HDRI_SUN_ELEV + 1.0)
    az = math.radians(sun_azimuth_deg)
    sun_dir = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    ld = bpy.data.lights.new("Sun", "SUN"); ld.energy = 7.0; ld.angle = math.radians(0.55)
    ld.color = (1.0, 0.63, 0.37)
    sun = bpy.data.objects.new("Sun", ld); scene.collection.objects.link(sun)
    from mathutils import Vector
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = Vector(tuple(-sun_dir)).to_track_quat("-Z", "Y")
    scene["sun_dir"] = sun_dir.tolist()
    return sun, sun_dir


def build_haze(coll, layers=((0, 24, 0.00062), (24, 900, 0.000085)),
               extent=((-5000, 9000), (-5000, 6000)), color=(0.85, 0.86, 0.90)):
    """Stacked homogeneous scattering layers (analytic in Cycles): exponential-like haze + valley mist."""
    objs = []
    for i, (z0, z1, dens) in enumerate(layers):
        (x0, x1), (y0, y1) = extent
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        ob = new_mesh_object(f"Haze{i}", np.array(v, float), f, coll, smooth=False)
        m = bpy.data.materials.new(f"Haze{i}")
        m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        vs = nt.nodes.new("ShaderNodeVolumeScatter"); vs.inputs["Density"].default_value = dens
        vs.inputs["Color"].default_value = (*color, 1); vs.inputs["Anisotropy"].default_value = 0.55
        L(vs.outputs[0], out.inputs["Volume"])
        try:
            m.cycles.homogeneous_volume = True
        except Exception:
            pass
        ob.data.materials.append(m)
        ob.visible_shadow = True
        objs.append(ob)
    return objs
