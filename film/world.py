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
CORRIDOR_S0 = 3300.0      # dense corridor terrain only where the earthworks are (viaduct section uses the far grid)

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
    """Radius of tower i at azimuth ang and relative height zf (0 base .. 1 top): irregular lobes that twist with
    height, buttresses, and vertical solution flutes (fenglin tower karst is jagged, not a smooth dome)."""
    return R * (1 + 0.22 * np.sin(3 * ang + i + 2.0 * zf) + 0.13 * np.sin(5 * ang + 2 * i - 3 * zf)
                + 0.08 * np.sin(9 * ang + 3 * i + 5 * zf) + 0.05 * np.sin(17 * ang + i + 9 * zf)
                + 0.035 * np.sin(29 * ang + 2 * i + 4 * zf) + 0.02 * np.sin(47 * ang + i)
                + 0.06 * np.sin(7 * ang + 5 * i + 11 * zf) * zf
                # v3: sharp V-shaped solution flutes and stepped bedding ledges / overhangs break the sculpted look
                - 0.05 * np.abs(np.sin(23 * ang + i + 3 * zf)) - 0.03 * np.abs(np.sin(61 * ang + 2 * i))
                + 0.045 * (np.mod(zf * 7.0 + 0.3 * np.sin(2 * ang + i), 1.0) - 0.5))


def tower_profile(i, d):
    """Height fraction vs normalised radius: sugar-loaf (steep flanks, rounded crown), varied per tower."""
    k = 3.0 + 1.6 * ((i * 37) % 7) / 6.0                  # steep, near-vertical lower walls
    return np.clip(1 - d ** k, 0, 1) ** (0.42 + 0.22 * ((i * 13) % 5) / 4.0)


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
    # v3: break the smooth dome skyline into karst knobs and saddles (only over the summit, d < 420-520 m, i.e.
    # behind the portal: the cutting, the portal face and the tunnel cover keep their v2 shape)
    core = np.clip((520 - d) / 100, 0, 1)
    massif = massif + core * (55 * np.clip(_noise2(x, y, 170, 7) + 0.1, 0, None) ** 1.4 + 18 * _noise2(x, y, 75, 8))
    # foothills rising above rail level from ~s=3850 so the line runs in a cutting before the tunnel
    rise = np.clip((x - 3450) / 550, 0, 1) ** 1.3 * 44 * (1 + 0.25 * _noise2(x, y, 250, 4))
    return 1.2 + valley + rise + river_depression(x, y) + massif


def formation(s, lat, g):
    """Apply railway earthworks: fill (embankment) and cut (cutting) around the line where not on viaduct."""
    form_z = A.RAIL_Z - 0.85                 # formation level below rail
    half = 7.0                               # formation half width (double track + cess + ditch)
    out = g.copy()
    on = (s > A.VIADUCT[1] - 20) & (s < A.TUNNEL_S + 80)
    al = np.abs(lat)
    # cutting slope 1:1 (soil) / embankment slope 1:1.5
    ex = np.maximum(al - half, 0)
    face = np.maximum(s - (A.TUNNEL_S - 14.0), 0.0)          # 1:1 portal face above the tunnel
    form_z = form_z + face
    tier = np.floor(ex / 10.0)                       # 1:1 slope, 2 m bench every 8 m of height
    cut = form_z + tier * 8.0 + np.minimum(ex - tier * 10.0, 8.0)
    fill = form_z - np.maximum(al - half, 0) / 1.5
    target = np.where(g > form_z, np.minimum(g, cut), np.maximum(g, fill))
    # ditches at the cutting toe
    ditch = (al > half - 1.2) & (al < half - 0.3) & (g > form_z)
    target = np.where(ditch, target - 0.5, target)
    w = np.clip((s - (A.VIADUCT[1] - 20)) / 30, 0, 1) * np.clip((A.TUNNEL_S + 80 - s) / 20, 0, 1)
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
    xs = np.concatenate([np.arange(-4000, -300, 30.0), np.arange(-300, 3600, 12.0), np.arange(3600, 8000.1, 30.0)])
    ys = np.concatenate([np.arange(-4500, -900, 30.0), np.arange(-900, 900, 12.0), np.arange(900, 5000.1, 30.0)])
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = natural_ground(X, Y)
    s, lat = A.nearest_s(X.ravel(), Y.ravel())
    s = s.reshape(X.shape); lat = lat.reshape(X.shape)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    ny, nx = X.shape
    faces = []
    inside = (np.abs(lat) < 150) & (s > CORRIDOR_S0 + 10) & (s < A.S_MAX - 10)
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
    ss = np.arange(CORRIDOR_S0, A.S_MAX - 5, 2.0)
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
            # leave the tunnel bore open: the lining tube (outer radius 7.5 m) closes the hole
            if A.TUNNEL_S - 14.0 < ss[i] < A.TUNNEL_S - 3.0 and abs(lats[j]) < 7.3 and abs(lats[j + 1]) < 7.3:
                continue
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
    cutm = ((nat > V[:, 2] + 0.3) & (np.abs(L2d) > 7.2) & (S2d > A.VIADUCT[1]) & (S2d < A.TUNNEL_S + 5)
            & (V[:, 2] < A.RAIL_Z + 15.0)).astype(np.float32)
    at = strip.data.attributes.new("cutmask", "FLOAT", "POINT"); at.data.foreach_set("value", cutm)
    cutz = ((nat > V[:, 2] + 0.3) & (np.abs(L2d) > 7.2) & (S2d > A.VIADUCT[1]) & (S2d < A.TUNNEL_S + 30)).astype(np.float32)
    az = strip.data.attributes.new("cutzone", "FLOAT", "POINT"); az.data.foreach_set("value", cutz)
    strip.data.materials.append(mats["terrain"])
    return far, strip


def build_towers(coll, mats):
    """Each tower: polar-grid sugar-loaf with height-varying lobes, ledges and overhang-free noise."""
    objs = []
    for i, (tx, ty, H, R) in enumerate(TOWERS):
        nr, na = 72, 256
        r = np.linspace(0, 1.3, nr) ** 0.8
        a = np.linspace(0, 2 * np.pi, na, endpoint=False)
        RR, AA = np.meshgrid(r, a, indexing="ij")
        prof = tower_profile(i, RR)
        zf = prof
        rr = _tower_r(i, AA, zf, R)
        X = tx + RR * rr * np.cos(AA); Y = ty + RR * rr * np.sin(AA)
        ledges = 0.03 * H * np.sin(RR * 31 + AA * 3 + i) * (RR < 1) * (prof > 0.15)
        bumps = (0.06 * H * _noise2(X, Y, 45, 100 + i) + 0.025 * H * _noise2(X, Y, 14, 200 + i)) * (RR < 1.05)
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
    rice.color_ramp.elements[0].color = (0.055, 0.12, 0.020, 1); rice.color_ramp.elements[1].color = (0.165, 0.225, 0.05, 1)   # v3: less saturated, a touch riper
    rice.color_ramp.elements.new(0.35).color = (0.07, 0.18, 0.02, 1)
    rice.color_ramp.elements.new(0.70).color = (0.10, 0.21, 0.03, 1)
    L(field_var, rice.inputs[0])
    ripe = nt.nodes.new("ShaderNodeMix"); ripe.data_type = "RGBA"; ripe.inputs["B"].default_value = (0.14, 0.19, 0.035, 1)
    L(is_ripe, ripe.inputs["Factor"]); L(rice.outputs[0], ripe.inputs["A"])
    fallow = nt.nodes.new("ShaderNodeMix"); fallow.data_type = "RGBA"
    L(is_fallow, fallow.inputs["Factor"]); L(ripe.outputs["Result"], fallow.inputs["A"]); L(mud.outputs["Color"], fallow.inputs["B"])
    riceN = nt.nodes.new("ShaderNodeTexNoise"); riceN.inputs["Scale"].default_value = 3.0; riceN.inputs["Detail"].default_value = 2
    L(geo.outputs["Position"], riceN.inputs["Vector"])
    ricem0 = nt.nodes.new("ShaderNodeMix"); ricem0.data_type = "RGBA"; ricem0.blend_type = "MULTIPLY"; ricem0.inputs["Factor"].default_value = 0.6
    L(fallow.outputs["Result"], ricem0.inputs["A"]); L(riceN.outputs["Color"], ricem0.inputs["B"])
    # uneven growth inside each field (10-50 m patches): darker/lighter, a touch yellower where thinner
    mott = nt.nodes.new("ShaderNodeTexNoise"); mott.inputs["Scale"].default_value = 0.045; mott.inputs["Detail"].default_value = 9
    mott.inputs["Roughness"].default_value = 0.68          # v3: down to metre-scale variation, not a flat 20 m wash
    L(geo.outputs["Position"], mott.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs[1].default_value = 0.3; mr.inputs[2].default_value = 0.7
    mr.inputs[3].default_value = 0.72; mr.inputs[4].default_value = 1.25; L(mott.outputs["Fac"], mr.inputs[0])
    ricem = nt.nodes.new("ShaderNodeMix"); ricem.data_type = "RGBA"; ricem.blend_type = "MULTIPLY"; ricem.inputs["Factor"].default_value = 1.0
    L(ricem0.outputs["Result"], ricem.inputs["A"])
    mcol = nt.nodes.new("ShaderNodeCombineColor"); L(mr.outputs[0], mcol.inputs[0]); L(mr.outputs[0], mcol.inputs[1])
    mb = nt.nodes.new("ShaderNodeMath"); mb.operation = "POWER"; mb.inputs[1].default_value = 1.6; L(mr.outputs[0], mb.inputs[0])
    L(mb.outputs[0], mcol.inputs[2]); L(mcol.outputs[0], ricem.inputs["B"])
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
    pad.inputs["B"].default_value = (0.13, 0.17, 0.06, 1)
    L(dk.outputs[0], pad.inputs["Factor"]); L(ricem.outputs["Result"], pad.inputs["A"])
    allc = nt.nodes.new("ShaderNodeMix"); allc.data_type = "RGBA"
    L(pz2.outputs[0], allc.inputs["Factor"]); L(rk.outputs["Result"], allc.inputs["A"]); L(pad.outputs["Result"], allc.inputs["B"])
    # diamond lattice concrete frames (slope protection) on cut slopes: uses UV = (chainage, lateral)
    uvn = nt.nodes.new("ShaderNodeUVMap")
    usep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(uvn.outputs[0], usep.inputs[0])
    def diag(sign):
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY_ADD"; a.inputs[1].default_value = sign
        L(usep.outputs["Y"], a.inputs[0]); L(usep.outputs["X"], a.inputs[2])
        b = nt.nodes.new("ShaderNodeMath"); b.operation = "PINGPONG"; b.inputs[1].default_value = 2.5; L(a.outputs[0], b.inputs[0])
        c = nt.nodes.new("ShaderNodeMath"); c.operation = "LESS_THAN"; c.inputs[1].default_value = 0.25; L(b.outputs[0], c.inputs[0])
        return c.outputs[0]
    lat1, lat2 = diag(1.0), diag(-1.0)
    latt = nt.nodes.new("ShaderNodeMath"); latt.operation = "MAXIMUM"; L(lat1, latt.inputs[0]); L(lat2, latt.inputs[1])
    cm = nt.nodes.new("ShaderNodeAttribute"); cm.attribute_name = "cutmask"
    lm = nt.nodes.new("ShaderNodeMath"); lm.operation = "MULTIPLY"; L(latt.outputs[0], lm.inputs[0])
    lm.inputs[1].default_value = 0.0      # v3: the lattice is real geometry now (infra.build_cutting_details)
    cz = nt.nodes.new("ShaderNodeAttribute"); cz.attribute_name = "cutzone"
    hydro = nt.nodes.new("ShaderNodeMix"); hydro.data_type = "RGBA"; hydro.blend_type = "MULTIPLY"; hydro.inputs["Factor"].default_value = 1.0
    hydro.inputs["B"].default_value = (0.55, 0.85, 0.24, 1)                   # hydroseeded slope grass: lush and bright
    L(gbw.outputs[0], hydro.inputs["A"])
    cutgrass = nt.nodes.new("ShaderNodeMix"); cutgrass.data_type = "RGBA"; L(cz.outputs["Fac"], cutgrass.inputs["Factor"])
    L(allc.outputs["Result"], cutgrass.inputs["A"]); L(hydro.outputs["Result"], cutgrass.inputs["B"])
    fcon = nt.nodes.new("ShaderNodeMix"); fcon.data_type = "RGBA"; fcon.blend_type = "MULTIPLY"; fcon.inputs["Factor"].default_value = 1.0
    fcon.inputs["B"].default_value = (0.62, 0.61, 0.58, 1)                  # weathered precast concrete beams
    L(rock.outputs["Color"], fcon.inputs["A"])
    frame_col = nt.nodes.new("ShaderNodeMix"); frame_col.data_type = "RGBA"; L(fcon.outputs["Result"], frame_col.inputs["B"])
    L(lm.outputs[0], frame_col.inputs["Factor"]); L(cutgrass.outputs["Result"], frame_col.inputs["A"])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(frame_col.outputs["Result"], p.inputs["Base Color"]); p.inputs["Roughness"].default_value = 0.85
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
    L(nor.outputs["Color"], nm.inputs["Color"])
    # raised bunds between paddies + raised concrete lattice beams on the cut slopes (bump on top of the grass normal)
    bh = nt.nodes.new("ShaderNodeMath"); bh.operation = "MULTIPLY"; L(dk.outputs[0], bh.inputs[0]); L(pz2.outputs[0], bh.inputs[1])
    bh2 = nt.nodes.new("ShaderNodeMath"); bh2.operation = "MULTIPLY_ADD"; bh2.inputs[1].default_value = 1.0
    L(bh.outputs[0], bh2.inputs[0]); L(lm.outputs[0], bh2.inputs[2])
    bbump = nt.nodes.new("ShaderNodeBump"); bbump.inputs["Strength"].default_value = 0.8; bbump.inputs["Distance"].default_value = 0.35
    L(bh2.outputs[0], bbump.inputs["Height"]); L(nm.outputs[0], bbump.inputs["Normal"]); L(bbump.outputs[0], p.inputs["Normal"])
    # water in flooded paddies: mirror-like with slight ripples, muddy tint
    wat = nt.nodes.new("ShaderNodeBsdfPrincipled")
    wat.inputs["Base Color"].default_value = (0.04, 0.05, 0.025, 1); wat.inputs["Roughness"].default_value = 0.30
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
    wfac = nt.nodes.new("ShaderNodeMath"); wfac.operation = "MULTIPLY"; wfac.inputs[1].default_value = 0.55
    L(wmask.outputs[0], wfac.inputs[0])
    mix = nt.nodes.new("ShaderNodeMixShader"); L(wfac.outputs[0], mix.inputs[0]); L(p.outputs[0], mix.inputs[1]); L(wat.outputs[0], mix.inputs[2])
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
    """Paddy layout: ~200 m blocks (coarse Voronoi), each with its own orientation, subdivided into parallel
    field strips 10-28 m wide and 25-70 m long.  R = per-field variation, G = dike mask, B = field type
    (0 rice, .33 ripening, .66 flooded, 1 fallow), A = 1."""
    from PIL import Image, ImageFilter
    from scipy.spatial import cKDTree
    W = int(PADDY_W / res); H = int(PADDY_H / res)
    rng = np.random.default_rng(seed)
    nb = int(PADDY_W * PADDY_H / (210.0 * 170.0))
    bseeds = np.column_stack([rng.uniform(0, W, nb), rng.uniform(0, H, nb)])
    tree = cKDTree(bseeds)
    bang = rng.normal(np.radians(8), np.radians(25), nb)
    bw = rng.uniform(10, 28, nb); bl = rng.uniform(25, 70, nb)
    R = np.zeros((H, W), np.uint8); G = np.zeros((H, W), np.uint8); B = np.zeros((H, W), np.uint8)
    types = np.array([0.0, 0.33, 0.66, 1.0]); probs = [0.74, 0.20, 0.04, 0.02]
    for y0 in range(0, H, 400):
        ys = np.arange(y0, min(H, y0 + 400))
        xx, yy = np.meshgrid(np.arange(W) + 0.5, ys + 0.5)
        wx = xx + 9 * np.sin(yy / 61.0) + 4 * np.sin(xx / 37.0)
        wy = yy + 8 * np.sin(xx / 53.0) + 3 * np.sin(yy / 29.0)
        d, idx = tree.query(np.column_stack([wx.ravel(), wy.ravel()]), k=2)
        b = idx[:, 0]
        block_edge = (d[:, 1] - d[:, 0]) < 1.2
        ca, sa = np.cos(bang[b]), np.sin(bang[b])
        u = wx.ravel() * ca + wy.ravel() * sa
        v = -wx.ravel() * sa + wy.ravel() * ca
        row = np.floor(v / bw[b]); fu = u + row * 17.3            # stagger field ends between rows
        col = np.floor(fu / bl[b])
        fv = v / bw[b] - row; fu2 = fu / bl[b] - col
        edge = block_edge | (np.minimum(fv, 1 - fv) * bw[b] < 0.32) | (np.minimum(fu2, 1 - fu2) * bl[b] < 0.32)
        h = (b * 73856093 ^ (row.astype(np.int64) * 19349663) ^ (col.astype(np.int64) * 83492791)) & 0xFFFF
        var = (h % 997) / 996.0
        tsel = np.searchsorted(np.cumsum(probs), (h % 1009) / 1008.0)
        R[ys] = (var * 255).reshape(len(ys), W)
        B[ys] = (types[np.clip(tsel, 0, 3)] * 255).reshape(len(ys), W)
        G[ys] = (edge * 255).reshape(len(ys), W)
    g = Image.fromarray(G).filter(ImageFilter.GaussianBlur(0.45))
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
            V.append((px, py, -0.6))
    faces = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(len(t) - 1)]
    faces = [(a, d, c, b) for (a, b, c, d) in faces]      # counter-clockwise from above -> normals up
    ob = new_mesh_object("River", np.array(V), faces, coll, smooth=True)
    if np.mean([p.normal.z for p in ob.data.polygons]) < 0:
        ob.data.flip_normals()
    m = bpy.data.materials.new("RiverWater")
    m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.020, 0.055, 0.045, 1); p.inputs["Roughness"].default_value = 0.04   # green karst river
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
    P = np.asarray(pts_xyz, float)
    si, li = A.nearest_s(P[:, 0], P[:, 1])
    for nm, arr in (("s_i", si), ("lat_i", li), ("hd_i", A.heading(si))):
        at = me.attributes.new(nm, "FLOAT", "POINT"); at.data.foreach_set("value", np.asarray(arr, np.float32))
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
    # ---- slipstream / wake tilt: grass bends downstream while the train passes and in its wake
    for nm, dv in (("Head S", -1e6), ("Tail S", -1e6), ("Wake", 0.0), ("Track Off", 2.5)):
        it = ng.interface.new_socket(nm, in_out="INPUT", socket_type="NodeSocketFloat"); it.default_value = dv
    def attr(nm):
        a = N.new("GeometryNodeInputNamedAttribute"); a.data_type = "FLOAT"; a.inputs["Name"].default_value = nm
        return a.outputs["Attribute"]
    def m(op, a, b=None, c=None):
        n = N.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]
    gi2 = N.new("NodeGroupInput")
    si, li, hd = attr("s_i"), attr("lat_i"), attr("hd_i")
    dtail = m("SUBTRACT", gi2.outputs["Tail S"], si)                       # >0: tail has passed this plant
    behind = m("GREATER_THAN", dtail, 0.0)
    beside = m("MULTIPLY", m("LESS_THAN", si, gi2.outputs["Head S"]), m("SUBTRACT", 1.0, behind))
    # time since the tail passed, clamped at 0: for plants the train never reaches, exp(-t/1.1) would be exp(+huge) = inf
    # and 0 * inf = NaN made every instance transform NaN (tower crowns, valley trees and rice were never rendered)
    tsince = m("MAXIMUM", m("DIVIDE", dtail, S.SPEED), 0.0)
    wdecay = m("EXPONENT", m("MULTIPLY", tsince, -1.0 / 1.1))
    timef = m("MULTIPLY_ADD", beside, 0.65, m("MULTIPLY", behind, wdecay))
    latd = m("MAXIMUM", m("SUBTRACT", m("ABSOLUTE", m("SUBTRACT", li, gi2.outputs["Track Off"])), 1.7), 0.0)
    latf = m("EXPONENT", m("MULTIPLY", latd, -1.0 / 2.5))
    flutter = N.new("ShaderNodeTexNoise"); flutter.noise_dimensions = "4D"; flutter.inputs["Scale"].default_value = 2.5
    ft = m("MULTIPLY", tm.outputs["Seconds"], 9.0); L(pos.outputs[0], flutter.inputs["Vector"]); L(ft, flutter.inputs["W"])
    amp = m("MULTIPLY", m("MULTIPLY", m("MULTIPLY", timef, latf), gi2.outputs["Wake"]), m("MULTIPLY_ADD", flutter.outputs["Fac"], 1.2, 0.4))
    # tilt towards +s: rotation vector (-sin hd, cos hd) * amp
    wx = m("ADD", sepw.outputs["X"], m("MULTIPLY", m("MULTIPLY", m("SINE", hd), -1.0), amp))
    wy = m("ADD", sepw.outputs["Y"], m("MULTIPLY", m("COSINE", hd), amp))
    comb2 = N.new("ShaderNodeCombineXYZ"); L(wx, comb2.inputs["X"]); L(wy, comb2.inputs["Y"]); L(yaw.outputs["Attribute"], comb2.inputs["Z"])
    e2r = N.new("FunctionNodeEulerToRotation"); L(comb2.outputs[0], e2r.inputs["Euler"])
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True
    L(gi.outputs["Geometry"], iop.inputs["Points"]); L(ci.outputs[0], iop.inputs["Instance"])
    L(pick.outputs["Attribute"], iop.inputs["Instance Index"]); L(e2r.outputs[0], iop.inputs["Rotation"])
    L(scl.outputs["Attribute"], iop.inputs["Scale"])
    L(iop.outputs[0], go.inputs["Geometry"])
    return ng


def _leaf_material(name, c0, c1, vor_scale, bump, sss):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vo = nt.nodes.new("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = vor_scale
    L(tc.outputs["Object"], vo.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.elements[0].color = (*c0, 1); cr.color_ramp.elements[1].color = (*c1, 1)
    L(vo.outputs["Distance"], cr.inputs[0])
    # per-tree colour: every instance gets its own green (Object Info > Random), like a real mixed grove
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    tint = nt.nodes.new("ShaderNodeValToRGB")
    e = tint.color_ramp.elements
    e[0].position = 0.0; e[0].color = (0.80, 0.92, 0.75, 1); e[1].position = 1.0; e[1].color = (1.22, 1.08, 0.80, 1)
    mid = e.new(0.5); mid.color = (1.0, 1.0, 1.0, 1)
    if name == "CrownLeaves":            # late October: a few Chinese tallow / early-turning trees among the greens
        e[1].position = 0.93
        t1 = e.new(0.935); t1.color = (3.2, 1.6, 0.55, 1)            # amber-yellow
        t2 = e.new(0.97); t2.color = (4.0, 0.9, 0.35, 1)             # tallow red
    L(oi.outputs["Random"], tint.inputs[0])
    mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs["Factor"].default_value = 1.0
    L(cr.outputs[0], mul.inputs["A"]); L(tint.outputs[0], mul.inputs["B"]); L(mul.outputs["Result"], p.inputs["Base Color"])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 1.0; bp.inputs["Distance"].default_value = bump
    L(vo.outputs["Distance"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.75
    p.inputs["Subsurface Weight"].default_value = sss
    p.inputs["Subsurface Radius"].default_value = (0.3, 0.6, 0.1)
    return m


def _bark():
    bark = bpy.data.materials.get("CrownBark")
    if bark is None:
        from . import materials as M
        bark = M.principled("CrownBark", (0.045, 0.035, 0.025), 0.0, 0.85)
    return bark


def _finish_proxy(name, coll, bm, leaf, trunk_test):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.materials.append(leaf); ob.data.materials.append(_bark())
    zc = np.zeros(len(ob.data.polygons) * 3); ob.data.polygons.foreach_get("center", zc)
    zc = zc.reshape(-1, 3)
    ob.data.polygons.foreach_set("material_index", trunk_test(zc).astype(np.int32))
    return ob


def _lobe(bm, rng, cx, cy, cz, r, subdiv=2, squash=0.85, rough=0.12):
    import bmesh
    ico = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    ph = rng.uniform(0, 6, 3)
    for v in ico["verts"]:
        p = np.array(v.co) / r
        n = 1 + rough * math.sin(4.0 * p[0] + ph[0]) * math.cos(3.5 * p[1] + ph[1]) \
            + 0.7 * rough * math.sin(7.0 * p[2] + ph[2]) + rng.uniform(-0.5, 0.5) * rough
        v.co = (cx + p[0] * r * n, cy + p[1] * r * n, cz + p[2] * r * n * squash)


def crown_proxy(name, coll, seed=0):
    """Mid-distance broadleaf tree (camphor / banyan type canopy): a cauliflower crown of 12-18 overlapping
    displaced lobes of different sizes over a main mass (irregular silhouette, no lollipop sphere), crown base low
    on the trunk, leafy shader with per-tree colour (no alpha cards)."""
    import bmesh
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    _lobe(bm, rng, 0.0, 0.0, 1.5, 0.72, subdiv=3, rough=0.10)
    for k in range(int(rng.integers(12, 19))):
        a = rng.uniform(0, 2 * math.pi); rad = rng.uniform(0.3, 0.85)
        top = rng.random() < 0.45
        cz = 1.5 + (rng.uniform(0.35, 0.85) if top else rng.uniform(-0.35, 0.4)) * (1 - 0.35 * rad)
        _lobe(bm, rng, rad * math.cos(a), rad * math.sin(a), cz, rng.uniform(0.26, 0.52), subdiv=2, rough=0.16)
    tr = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.13, radius2=0.07, depth=1.6)
    for v in tr["verts"]:
        v.co.z += 0.8
    leaf = _leaf_material("CrownLeaves", (0.02, 0.05, 0.012), (0.11, 0.19, 0.04), 9.0, 0.25, 0.15)
    return _finish_proxy(name, coll, bm, leaf, lambda zc: (np.hypot(zc[:, 0], zc[:, 1]) < 0.16) & (zc[:, 2] < 1.2))


def bamboo_proxy(name, coll, seed=0):
    """Clumping bamboo (Bambusa, the feathery groves along Guangxi rivers and around villages): 16-24 culms rise
    from one root clump, arch outwards and droop at the tips; the fine foliage is a chain of small soft lobes along
    the upper two thirds of every culm, giving the tall fountain-shaped, feathery mass seen from a distance."""
    import bmesh
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    H = 2.9
    for k in range(int(rng.integers(16, 25))):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.35, 1.05)                      # how far the culm arches outwards (x H)
        h = H * rng.uniform(0.75, 1.0)
        b0 = np.array([0.18 * math.cos(a), 0.18 * math.sin(a), 0.0])
        out = np.array([math.cos(a), math.sin(a), 0.0])
        def P(t):                                           # quadratic arch: up, outwards, tip drooping
            return b0 + out * lean * h * t ** 1.6 + np.array([0, 0, h * (1.15 * t - 0.25 * t ** 3)])
        pts = [P(t) for t in np.linspace(0, 1, 8)]
        for p0, p1 in zip(pts[:3], pts[1:4]):              # bare lower culm
            cul = bmesh.ops.create_cone(bm, cap_ends=False, segments=5, radius1=0.03, radius2=0.025,
                                        depth=float(np.linalg.norm(p1 - p0)))
            d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
            u = np.cross(d, (0, 0, 1.0)); u = u / (np.linalg.norm(u) + 1e-9) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
            w = np.cross(d, u); c = (p0 + p1) / 2
            for v in cul["verts"]:
                q = np.array(v.co); v.co = tuple(c + q[0] * u + q[1] * w + q[2] * d)
        for t in np.linspace(0.18, 1.0, int(rng.integers(11, 16))):
            c = P(t) + rng.normal(0, 0.07, 3)
            _lobe(bm, rng, c[0], c[1], c[2], 0.13 + 0.13 * math.sin(math.pi * min(t, 0.95)) * rng.uniform(0.8, 1.2),
                  subdiv=1, squash=0.8, rough=0.22)
    leaf = _leaf_material("BambooLeaves", (0.05, 0.10, 0.015), (0.20, 0.27, 0.05), 22.0, 0.08, 0.25)
    return _finish_proxy(name, coll, bm, leaf, lambda zc: np.zeros(len(zc), bool))


def sample_surface(obj_list, density, rng, filt=None, max_n=200000, tri_filter=None):
    """Random points on the evaluated surfaces of objects (area weighted), optional filter f(p, n).
    tri_filter(centroids) restricts the triangles BEFORE sampling, so max_n is spent inside the region
    (filtering afterwards wasted ~98 % of the points when the region is small compared with the mesh)."""
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
        if tri_filter is not None:
            sel = tri_filter((a + b + c) / 3.0)
            a, b, c, nrm, area = a[sel], b[sel], c[sel], nrm[sel], area[sel]
            tri = tri[sel]
            if len(area) == 0:
                continue
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
def build_sky(scene, hdri_path, sun_azimuth_deg=262.0, exposure=0.0, strength=0.70, hdri_sun=(0.612, 4.64),
              sun_color=(1.0, 0.60, 0.33), sun_energy=13.0):
    """HDRI sky rotated so its (clipped) sun lies at the requested azimuth, plus a matched warm sun lamp.
    hdri_sun = (u of the sun disc in the equirect image, its elevation in degrees), measured from the image."""
    HDRI_SUN_U, HDRI_SUN_ELEV = hdri_sun
    phi0 = (HDRI_SUN_U - 0.5) * 2 * math.pi
    d0 = np.array([math.cos(phi0), -math.sin(phi0)])            # Blender equirect: u = -atan2(y, x)/2pi + 0.5
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
    el = math.radians(HDRI_SUN_ELEV + (1.0 if HDRI_SUN_ELEV < 10 else 0.0))
    az = math.radians(sun_azimuth_deg)
    sun_dir = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    ld = bpy.data.lights.new("Sun", "SUN"); ld.energy = sun_energy; ld.angle = math.radians(0.55)
    ld.color = sun_color
    sun = bpy.data.objects.new("Sun", ld); scene.collection.objects.link(sun)
    from mathutils import Vector
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = Vector(tuple(-sun_dir)).to_track_quat("-Z", "Y")
    scene["sun_dir"] = sun_dir.tolist()
    return sun, sun_dir


def build_haze(coll, layers=((-12, 24, 0.00062), (24, 900, 0.000085)),
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


def build_village(coll, assets, n=26, centre_s=2700.0, lat=-300.0, seed=4, spread_s=260.0, spread_l=(-120.0, 90.0),
                  name="Village", hamlet=False):
    """Rural Guangxi village from the house kit in film/village.py (five building types, real openings,
    balconies, tiled / pantile roofs, solar water heaters, walled yards).  Houses stand 17 m apart or more,
    face the village street (either side), and avoid ground that slopes more than 3 m across a house."""
    from . import village as V
    rng = np.random.default_rng(seed)
    yaw_v = A.heading(centre_s) + rng.choice([0.0, math.pi / 2]) + rng.uniform(-0.3, 0.3)
    sites = []
    tries = 0
    while len(sites) < n and tries < 40 * n:
        tries += 1
        s = centre_s + rng.uniform(-spread_s, spread_s); l = lat + rng.uniform(*spread_l)
        th = A.heading(s); x0, y0 = A.xy(s)
        x = x0 - math.sin(th) * l; y = y0 + math.cos(th) * l
        if any(math.hypot(x - q[0], y - q[1]) < 17.0 for q in sites):
            continue
        if not _clear_of_features(np.array([x]), np.array([y]), 12.0)[0]:      # not in the river / on a tower foot
            continue
        cx = x + np.array([-7, 7, 7, -7, 0]); cy = y + np.array([-7, -7, 7, 7, 0])
        gz = natural_ground(cx, cy)
        if gz.max() - gz.min() > 3.0:
            continue
        yaw = yaw_v + rng.choice([0.0, math.pi]) + rng.normal(0, 0.06)
        sites.append((x, y, float(gz.min()), yaw))
    return V.place(coll, assets, sites, rng, hamlet=hamlet, name=name)


def _clear_of_features(x, y, margin=40.0, river=30.0):
    """True where (x, y) is on the open valley floor: away from karst towers, the river and the hills."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.ones(x.shape, bool)
    for i, (tx, ty, H, R) in enumerate(TOWERS):
        ok &= np.hypot(x - tx, y - ty) > 1.25 * R + margin
    ok &= np.abs(x - RIVER_X(y + 500)) > river
    ok &= natural_ground(x, y) < 6.0
    return ok


def hamlet_sites(n=10, seed=21):
    """Small farm hamlets spread over the valley floor (rail coordinates s, lat), clear of towers/river/line."""
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 5000:
        tries += 1
        s = rng.uniform(250, 3300); lat = rng.choice([-1, 1]) * rng.uniform(260, 1300)
        th = A.heading(s); x0, y0 = A.xy(s)
        x = x0 - math.sin(th) * lat; y = y0 + math.cos(th) * lat
        if not _clear_of_features(np.array([x]), np.array([y]), 70.0)[0]:
            continue
        if abs(s - 2700) < 420 and -480 < lat < -120:          # the existing village
            continue
        if any(math.hypot(x - q[2], y - q[3]) < 420 for q in out):
            continue
        out.append((s, lat, x, y))
    # v3: hamlets placed where the cameras actually look (found by projecting the valley floor into every shot):
    # S1 aerial, S2 between the piers, S3 drone both sides, S4 behind the train, S8 600 mm background, S9 far end
    for (fs, fl) in FEATURED_HAMLETS:
        th = A.heading(fs); x0, y0 = A.xy(fs)
        x = x0 - math.sin(th) * fl; y = y0 + math.cos(th) * fl
        out = [q for q in out if math.hypot(x - q[2], y - q[3]) > 260]
        out.append((fs, fl, x, y))
    return out


FEATURED_HAMLETS = [(1500, -400), (1230, -170), (2070, -130), (2130, 170), (2660, 140), (2550, -110), (3480, -120)]


def valley_tree_points(rng, hamlets, scale=1.0):
    """Tree clumps on the valley floor like real Guangxi farmland: copses, tree lines along the river,
    groves around hamlets and lone trees on field corners. Returns (N, 3) points."""
    pts = []
    def add(x, y):
        pts.append(np.column_stack([x, y]))
    # copses scattered over the paddies
    nc = int(900 * scale)
    cx = rng.uniform(-1400, 4400, nc * 3); cy = rng.uniform(-1700, 2300, nc * 3)
    k = _clear_of_features(cx, cy, 25.0); cx, cy = cx[k][:nc], cy[k][:nc]
    for x0, y0 in zip(cx, cy):
        m = int(rng.integers(8, 40)); r = rng.uniform(5, 15)
        add(x0 + rng.normal(0, r, m), y0 + rng.normal(0, r * rng.uniform(0.4, 1.0), m))
    # river banks
    t = np.arange(-1500, 2500, 5.5)
    xr = RIVER_X(t); yr = t - 500
    for sgn in (-1, 1):
        off = 22 + rng.uniform(0, 9, len(t))
        keep = rng.random(len(t)) < 0.6
        add((xr + sgn * off)[keep], yr[keep])
    # groves around hamlets (and the main village)
    th = float(A.heading(2700.0)); vx, vy = (float(v) for v in A.xy(2700.0))
    main_village = (2700.0, -300.0, vx + math.sin(th) * 300.0, vy - math.cos(th) * 300.0)
    for (s, lat, x, y) in list(hamlets) + [main_village]:
        m = int(rng.integers(30, 70))
        add(x + rng.normal(0, 55, m), y + rng.normal(0, 55, m))
    # lone trees on field corners
    nl = int(180 * scale)
    add(rng.uniform(-1400, 4400, nl), rng.uniform(-1700, 2300, nl))
    P = np.concatenate(pts)
    keep = _clear_of_features(P[:, 0], P[:, 1], 6.0, river=21.0)      # bank groves stand right at the water
    s, lat = A.nearest_s(P[:, 0], P[:, 1])
    keep &= np.abs(lat) > 22.0                                   # nothing growing on/next to the viaduct
    P = P[keep]
    Z = ground(P[:, 0], P[:, 1])
    return np.column_stack([P, Z])


def rice_clump(name, coll, seed=0):
    """One transplanted rice hill: ~18 arching blades 0.6-0.9 m, green with yellowing tips."""
    rng = np.random.default_rng(seed)
    verts, faces = [], []
    for b in range(18):
        az = rng.uniform(0, 2 * np.pi); lean = rng.uniform(0.15, 0.55); h = rng.uniform(0.6, 0.9)
        w = rng.uniform(0.008, 0.013); bend = rng.uniform(0.2, 0.6)
        base = len(verts)
        for k in range(6):
            f = k / 5
            r = lean * f * h + bend * f * f * 0.25
            z = h * (f - 0.35 * bend * f * f)
            cx, cy = r * math.cos(az), r * math.sin(az)
            ww = w * (1 - 0.85 * f)
            px, py = -math.sin(az) * ww, math.cos(az) * ww
            verts += [(cx - px, cy - py, z), (cx + px, cy + py, z)]
        for k in range(5):
            a = base + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))
    ob = new_mesh_object(name, np.array(verts), faces, coll, smooth=True)
    m = bpy.data.materials.get("RiceBlade")
    if m is None:
        m = bpy.data.materials.new("RiceBlade"); m.use_nodes = True
        nt = m.node_tree; p = nt.nodes["Principled BSDF"]; L = nt.links.new
        geo = nt.nodes.new("ShaderNodeNewGeometry"); sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        tc = nt.nodes.new("ShaderNodeTexCoord"); sp2 = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sp2.inputs[0])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.07, 0.16, 0.02, 1); ramp.color_ramp.elements[1].color = (0.42, 0.42, 0.10, 1)
        ramp.color_ramp.elements.new(0.65).color = (0.17, 0.32, 0.04, 1)
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs[1].default_value = 0.0; mr.inputs[2].default_value = 0.95
        L(sp2.outputs["Z"], mr.inputs[0]); L(mr.outputs[0], ramp.inputs[0])
        hs = nt.nodes.new("ShaderNodeHueSaturation"); L(ramp.outputs[0], hs.inputs["Color"])
        mv = nt.nodes.new("ShaderNodeMath"); mv.operation = "MULTIPLY_ADD"; mv.inputs[1].default_value = 0.3; mv.inputs[2].default_value = 0.85
        L(oi.outputs["Random"], mv.inputs[0]); L(mv.outputs[0], hs.inputs["Value"])
        L(hs.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.45; p.inputs["Subsurface Weight"].default_value = 0.0
        p.inputs["Transmission Weight"].default_value = 0.0
        p.inputs["Sheen Weight"].default_value = 0.2
    ob.data.materials.append(m)
    return ob


def scatter_rice(coll, src_coll, centre, radius=40.0, paddy_img_path=None, seed=9, keep_fn=None):
    """Rice hills in transplant rows (0.25 x 0.30 m) around a camera, skipping dikes."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    clumps = [rice_clump(f"Rice{i}", src_coll, i) for i in range(4)]
    for c in clumps:
        c.hide_render = True
    img = np.asarray(Image.open(paddy_img_path))[::-1] if paddy_img_path else None
    ang = math.radians(8)
    u = np.arange(-radius, radius, 0.25); v = np.arange(-radius, radius, 0.30)
    U, Vv = np.meshgrid(u, v)
    X = centre[0] + U * math.cos(ang) - Vv * math.sin(ang) + rng.normal(0, 0.03, U.shape)
    Y = centre[1] + U * math.sin(ang) + Vv * math.cos(ang) + rng.normal(0, 0.03, U.shape)
    keep = np.hypot(X - centre[0], Y - centre[1]) < radius
    keep &= np.hypot(X - centre[0], Y - centre[1]) > 1.2
    if img is not None:
        px = ((X - PADDY_X0)).astype(int).clip(0, img.shape[1] - 1); py = ((Y - PADDY_Y0)).astype(int).clip(0, img.shape[0] - 1)
        keep &= img[py, px, 1] < 60                                   # not on a dike
        keep &= img[py, px, 2] < 150                                  # rice / ripening fields only
    if keep_fn is not None:
        keep &= keep_fn(X, Y)
    X, Y = X[keep], Y[keep]
    Z = natural_ground(X, Y)
    pts = np.stack([X, Y, Z], -1)
    return scatter_tree("VegRice", coll, pts, clumps, (0.85, 1.15), seed, wind=0.10)


def build_mist_layers(coll, heights=(4.0, 5.2, 6.6, 8.2, 10.0, 12.0, 14.3, 16.9, 19.8, 23.0, 26.6, 30.6, 35.0), tau=0.011,
                      extent=((-4000, 8000), (-4500, 5000)), colour=(0.86, 0.80, 0.72), strength=0.95):
    """Valley mist as stacked translucent layers (cheap, noise-free alternative to volumes).

    Each layer's opacity uses the true path length through a slab: a = 1 - exp(-tau_eff / |cos(theta)|),
    with tau_eff modulated by drifting 4D noise (patchy, slowly moving mist) and fading with height.
    """
    m = bpy.data.materials.new("MistLayer")
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    L(geo.outputs["Normal"], dot.inputs[0]); L(geo.outputs["Incoming"], dot.inputs[1])
    ab = nt.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; L(dot.outputs["Value"], ab.inputs[0])
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; mx.inputs[1].default_value = 0.035; L(ab.outputs[0], mx.inputs[0])
    # patchy density: 4D noise in world space, drifting with time (frame driven W)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.noise_dimensions = "4D"; nz.inputs["Scale"].default_value = 0.0025
    nz.inputs["Detail"].default_value = 3; nz.inputs["Roughness"].default_value = 0.55
    L(geo.outputs["Position"], nz.inputs["Vector"])
    drv = nz.inputs["W"].driver_add("default_value")
    drv.driver.expression = "frame / 2400.0"
    dens = nt.nodes.new("ShaderNodeMapRange"); dens.inputs[1].default_value = 0.35; dens.inputs[2].default_value = 0.75
    dens.inputs[3].default_value = 0.15; dens.inputs[4].default_value = 1.6
    L(nz.outputs["Fac"], dens.inputs[0])
    # layer height falloff (object property 'hf') and tau
    hf = nt.nodes.new("ShaderNodeAttribute"); hf.attribute_type = "OBJECT"; hf.attribute_name = "hf"
    t1 = nt.nodes.new("ShaderNodeMath"); t1.operation = "MULTIPLY"; t1.inputs[1].default_value = tau
    L(dens.outputs[0], t1.inputs[0])
    t2 = nt.nodes.new("ShaderNodeMath"); t2.operation = "MULTIPLY"; L(t1.outputs[0], t2.inputs[0]); L(hf.outputs["Fac"], t2.inputs[1])
    dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"; L(t2.outputs[0], dv.inputs[0]); L(mx.outputs[0], dv.inputs[1])
    ng = nt.nodes.new("ShaderNodeMath"); ng.operation = "MULTIPLY"; ng.inputs[1].default_value = -1.0; L(dv.outputs[0], ng.inputs[0])
    ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"; L(ng.outputs[0], ex.inputs[0])
    al = nt.nodes.new("ShaderNodeMath"); al.operation = "SUBTRACT"; al.inputs[0].default_value = 1.0; L(ex.outputs[0], al.inputs[1])
    # brightness: sunlit mist glows towards the sun (Henyey-Greenstein-like boost from the scene sun direction)
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*colour, 1); em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader"); L(al.outputs[0], mix.inputs[0]); L(tr.outputs[0], mix.inputs[1]); L(em.outputs[0], mix.inputs[2])
    L(mix.outputs[0], out.inputs["Surface"])
    objs = []
    (x0, x1), (y0, y1) = extent
    for i, h in enumerate(heights):
        v = [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
        ob = new_mesh_object(f"Mist{i}", np.array(v, float), [(0, 1, 2, 3)], coll, smooth=False)
        ob.data.materials.append(m)
        ob["hf"] = float(math.exp(-(h - 4.0) / 12.0))
        ob.visible_shadow = False
        ob.visible_diffuse = False
        ob.visible_glossy = True
        objs.append(ob)
    return objs


# ------------------------------------------------------------------ v3: farm roads, roadside ditches, river banks
def _ribbon(name, coll, xs, ys, prof, mat, z_fn, uv_scale=1.0):
    """Ribbon draped on the ground along a polyline: prof = [(offset_m, dz_m), ...] across the path."""
    dx = np.gradient(xs); dy = np.gradient(ys); n = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / n, dx / n
    along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    V = []
    for k in range(len(xs)):
        for (o, dz) in prof:
            px, py = xs[k] + nx[k] * o, ys[k] + ny[k] * o
            V.append((px, py, 0.0, dz))
    V = np.array(V)
    V[:, 2] = z_fn(V[:, 0], V[:, 1]) + V[:, 3]
    m = len(prof)
    F, UV = [], []
    for k in range(len(xs) - 1):
        for j in range(m - 1):
            a = k * m + j
            F.append((a, a + 1, a + m + 1, a + m))
            UV += [(prof[j][0] * uv_scale, along[k] * uv_scale), (prof[j + 1][0] * uv_scale, along[k] * uv_scale),
                   (prof[j + 1][0] * uv_scale, along[k + 1] * uv_scale), (prof[j][0] * uv_scale, along[k + 1] * uv_scale)]
    ob = new_mesh_object(name, V[:, :3], F, coll, smooth=True, uvs=np.array(UV))
    ob.data.materials.append(mat)
    return ob


ROADS = []          # road centre polylines (x, y): the vegetation pass plants tree lines along them


def build_roads(coll, tex, hamlets):
    """Concrete village roads (3.5 m, the 'every village connected' standard) on a low embankment, linking the
    hamlets and the main village along a minimum spanning tree (no river crossings, around the towers), each with
    a water-filled irrigation ditch on one side.  Plus muddy, pebbly banks along both sides of the river."""
    from . import infra
    rng = np.random.default_rng(77)
    th = float(A.heading(2700.0)); vx, vy = (float(v) for v in A.xy(2700.0))
    nodes = [(vx + math.sin(th) * 300.0, vy - math.cos(th) * 300.0)] + [(h[2], h[3]) for h in hamlets]
    nodes = np.array(nodes)
    side_of_river = np.sign(nodes[:, 0] - RIVER_X(nodes[:, 1] + 500))
    # Prim's MST on distance, edges that cross the river or run into a tower are not allowed
    def path(a, b):
        L = float(np.hypot(*(nodes[b] - nodes[a])))
        t = np.linspace(0, 1, max(int(L / 4.0), 3))
        px = nodes[a][0] + (nodes[b][0] - nodes[a][0]) * t
        py = nodes[a][1] + (nodes[b][1] - nodes[a][1]) * t
        # gentle meander like a road following field boundaries
        nxp, nyp = -(nodes[b][1] - nodes[a][1]) / L, (nodes[b][0] - nodes[a][0]) / L
        amp = rng.uniform(15, 40) * np.sin(np.pi * t) * np.sin(2 * np.pi * t * rng.uniform(0.5, 1.5) + rng.uniform(0, 6))
        return px + nxp * amp, py + nyp * amp
    def ok(px, py):
        s, lat = A.nearest_s(px, py)
        good = _clear_of_features(px, py, 8.0, river=24.0) | (np.abs(lat) < 40)
        return bool(good.all())
    road_mat = infra.concrete("RoadConcrete", tex.get("concrete"), (0.92, 0.90, 0.86), 0.25, streaks=False)
    ditch_mat = bpy.data.materials.get("RiverWater")
    tree_ = {0}
    edges = []
    while len(tree_) < len(nodes):
        best = None
        for a in tree_:
            for b in range(len(nodes)):
                if b in tree_ or side_of_river[a] != side_of_river[b]:
                    continue
                d = float(np.hypot(*(nodes[b] - nodes[a])))
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:                          # the rest is across the river: start a second tree there
            rest = [b for b in range(len(nodes)) if b not in tree_]
            tree_.add(rest[0]); continue
        tree_.add(best[2]); edges.append((best[1], best[2]))
    obs = []
    prof = [(-2.9, -0.25), (-1.75, 0.18), (1.75, 0.18), (2.9, -0.25)]
    ditch = [(3.3, -0.05), (4.2, -0.05)]
    def zf(x, y):
        return natural_ground(x, y)
    for k, (a, b) in enumerate(edges):
        for attempt in range(4):
            px, py = path(a, b)
            if ok(px, py):
                break
        else:
            continue
        s, lat = A.nearest_s(px, py)
        keep = np.abs(lat) > 16.0                                     # pass under the viaduct, not through piers
        if keep.sum() < 3:
            continue
        obs.append(_ribbon(f"Road{k}", coll, px, py, prof, road_mat, zf, uv_scale=0.25))
        ROADS.append((px, py))
        if ditch_mat is not None:
            obs.append(_ribbon(f"RoadDitch{k}", coll, px, py, ditch, ditch_mat, zf))
    # river banks: mud and pebbles from just under the water line up the bank
    mud = infra.concrete("RiverBankMud", tex.get("mud"), (0.80, 0.74, 0.62), 0.5, streaks=False)
    t = np.linspace(-1500, 2500, 800)
    xr = RIVER_X(t); yr = t - 500
    width = 34 + 6 * np.sin(t / 200)
    for sgn in (-1, 1):
        dx = np.gradient(xr); dy = np.gradient(yr); n = np.hypot(dx, dy)
        nx, ny = -dy / n, dx / n
        xs = xr + nx * sgn * (width / 2); ys = yr + ny * sgn * (width / 2)
        pb = [(-sgn * 1.5, -0.35), (0.0, 0.03), (sgn * 1.8, 0.03), (sgn * 3.6, 0.02)]
        if sgn < 0:
            pb = pb[::-1]
        def zb(x, y):
            return np.maximum(natural_ground(x, y), -0.6)
        obs.append(_ribbon(f"RiverBank{sgn:+d}", coll, xs, ys, pb, mud, zb, uv_scale=0.5))
    print(f"[build]   roads: {len(edges)} links, river banks", flush=True)
    return obs
