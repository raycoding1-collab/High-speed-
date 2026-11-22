"""Shader networks (Cycles)."""
import bpy


def _nt(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return nt, out


def _img(nt, path, noncolor=False, interp="Cubic"):
    n = nt.nodes.new("ShaderNodeTexImage")
    im = bpy.data.images.load(path, check_existing=True)
    if noncolor:
        im.colorspace_settings.name = "Non-Color"
    im.alpha_mode = "CHANNEL_PACKED"      # alpha holds data (glass mask / roughness), keep RGB intact
    n.image = im
    n.interpolation = interp
    return n


def principled(name, color=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, coat=0.0, emission=None, estrength=0.0, alpha=1.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    nt, out = _nt(m)
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = 0.04
    if emission:
        p.inputs["Emission Color"].default_value = (*emission, 1)
        p.inputs["Emission Strength"].default_value = estrength
    p.inputs["Alpha"].default_value = alpha
    nt.links.new(p.outputs[0], out.inputs[0])
    return m


def body_material(name, col_path, dat_path):
    """Car body: livery albedo, metallic silver / painted areas with clear coat, panel-gap bump,
    road dirt, insect marks, glass where the mask says so, interior trim on back faces."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    nt, out = _nt(m)
    L = nt.links.new
    col = _img(nt, col_path); dat = _img(nt, dat_path, noncolor=True)
    sep = nt.nodes.new("ShaderNodeSeparateColor"); L(dat.outputs["Color"], sep.inputs[0])
    met, gap, dirt = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    # dirt colour mixed into albedo
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    mixc.inputs["B"].default_value = (0.16, 0.135, 0.10, 1)
    L(dirt, mixc.inputs["Factor"]); L(col.outputs["Color"], mixc.inputs["A"])
    # gap darkening
    gapd = nt.nodes.new("ShaderNodeMix"); gapd.data_type = "RGBA"
    gapd.inputs["B"].default_value = (0.01, 0.01, 0.012, 1)
    gm = nt.nodes.new("ShaderNodeMath"); gm.operation = "MULTIPLY"; gm.inputs[1].default_value = 0.85
    L(gap, gm.inputs[0]); L(gm.outputs[0], gapd.inputs["Factor"]); L(mixc.outputs["Result"], gapd.inputs["A"])
    # roughness = A + dirt*0.45
    rmath = nt.nodes.new("ShaderNodeMath"); rmath.operation = "MULTIPLY_ADD"
    rmath.inputs[1].default_value = 0.45
    L(dirt, rmath.inputs[0]); L(dat.outputs["Alpha"], rmath.inputs[2])
    # metallic reduced by dirt
    mm = nt.nodes.new("ShaderNodeMath"); mm.operation = "MULTIPLY"
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    L(dirt, inv.inputs[1]); L(met, mm.inputs[0]); L(inv.outputs[0], mm.inputs[1])
    # coat weight = (1 - dirt) * 0.9
    cw = nt.nodes.new("ShaderNodeMath"); cw.operation = "MULTIPLY"; cw.inputs[1].default_value = 0.55
    L(inv.outputs[0], cw.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.004
    ginv = nt.nodes.new("ShaderNodeMath"); ginv.operation = "SUBTRACT"; ginv.inputs[0].default_value = 1.0
    L(gap, ginv.inputs[1]); L(ginv.outputs[0], bump.inputs["Height"])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(gapd.outputs["Result"], p.inputs["Base Color"]); L(mm.outputs[0], p.inputs["Metallic"])
    L(rmath.outputs[0], p.inputs["Roughness"]); L(cw.outputs[0], p.inputs["Coat Weight"])
    p.inputs["Coat Roughness"].default_value = 0.06
    p.inputs["Specular IOR Level"].default_value = 0.35
    L(bump.outputs[0], p.inputs["Normal"]); L(bump.outputs[0], p.inputs["Coat Normal"])
    # glass: mask A. 1 = side window (dark tint), 0.5 windscreen (light green tint), 0.25 head/tail lamp lens
    a = col.outputs["Alpha"]
    glass_amt = nt.nodes.new("ShaderNodeMath"); glass_amt.operation = "GREATER_THAN"; glass_amt.inputs[1].default_value = 0.4
    L(a, glass_amt.inputs[0])
    is_ws = nt.nodes.new("ShaderNodeMath"); is_ws.operation = "COMPARE"; is_ws.inputs[1].default_value = 0.5; is_ws.inputs[2].default_value = 0.2
    L(a, is_ws.inputs[0])
    tint = nt.nodes.new("ShaderNodeMix"); tint.data_type = "RGBA"
    tint.inputs["A"].default_value = (0.05, 0.055, 0.055, 1)   # side windows: privacy tint (~5 % transmission)
    tint.inputs["B"].default_value = (0.62, 0.70, 0.66, 1)   # windscreen
    L(is_ws.outputs[0], tint.inputs["Factor"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent"); L(tint.outputs["Result"], tr.inputs[0])
    gl = nt.nodes.new("ShaderNodeBsdfGlossy"); gl.inputs["Roughness"].default_value = 0.01
    fr = nt.nodes.new("ShaderNodeFresnel")
    # thin pane: no total internal reflection when seen from inside -> undo the node's automatic 1/IOR on back faces
    geo0 = nt.nodes.new("ShaderNodeNewGeometry")
    ior = nt.nodes.new("ShaderNodeMath"); ior.operation = "MULTIPLY_ADD"; ior.inputs[1].default_value = 1 / 1.52 - 1.52; ior.inputs[2].default_value = 1.52
    L(geo0.outputs["Backfacing"], ior.inputs[0]); L(ior.outputs[0], fr.inputs["IOR"])
    gmix = nt.nodes.new("ShaderNodeMixShader"); L(fr.outputs[0], gmix.inputs[0]); L(tr.outputs[0], gmix.inputs[1]); L(gl.outputs[0], gmix.inputs[2])
    # lamp lens: emission driven by object property "lamp" (rgb * strength)
    is_lamp = nt.nodes.new("ShaderNodeMath"); is_lamp.operation = "COMPARE"; is_lamp.inputs[1].default_value = 0.25; is_lamp.inputs[2].default_value = 0.1
    L(a, is_lamp.inputs[0])
    lampattr = nt.nodes.new("ShaderNodeAttribute"); lampattr.attribute_type = "OBJECT"; lampattr.attribute_name = "lamp"
    em = nt.nodes.new("ShaderNodeEmission"); L(lampattr.outputs["Color"], em.inputs[0]); em.inputs[1].default_value = 1.0
    lens = nt.nodes.new("ShaderNodeAddShader"); L(em.outputs[0], lens.inputs[0]); L(gl.outputs[0], lens.inputs[1])
    s1 = nt.nodes.new("ShaderNodeMixShader"); L(glass_amt.outputs[0], s1.inputs[0]); L(p.outputs[0], s1.inputs[1]); L(gmix.outputs[0], s1.inputs[2])
    s2 = nt.nodes.new("ShaderNodeMixShader"); L(is_lamp.outputs[0], s2.inputs[0]); L(s1.outputs[0], s2.inputs[1]); L(lens.outputs[0], s2.inputs[2])
    # back faces (inside of the shell) -> interior wall trim, unless glass
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    inner = nt.nodes.new("ShaderNodeBsdfDiffuse"); inner.inputs[0].default_value = (0.55, 0.54, 0.50, 1)
    bf = nt.nodes.new("ShaderNodeMath"); bf.operation = "MULTIPLY"
    ng = nt.nodes.new("ShaderNodeMath"); ng.operation = "SUBTRACT"; ng.inputs[0].default_value = 1.0
    L(glass_amt.outputs[0], ng.inputs[1]); L(geo.outputs["Backfacing"], bf.inputs[0]); L(ng.outputs[0], bf.inputs[1])
    s3 = nt.nodes.new("ShaderNodeMixShader"); L(bf.outputs[0], s3.inputs[0]); L(s2.outputs[0], s3.inputs[1]); L(inner.outputs[0], s3.inputs[2])
    L(s3.outputs[0], out.inputs[0])
    return m


def gangway_material():
    m = bpy.data.materials.get("Gangway")
    if m:
        return m
    m = bpy.data.materials.new("Gangway")
    nt, out = _nt(m)
    L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sep.inputs[0])
    wave = nt.nodes.new("ShaderNodeMath"); wave.operation = "SINE"
    sc = nt.nodes.new("ShaderNodeMath"); sc.operation = "MULTIPLY"; sc.inputs[1].default_value = 2 * 3.14159 / 0.045
    L(sep.outputs["X"], sc.inputs[0]); L(sc.outputs[0], wave.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.5; bump.inputs["Distance"].default_value = 0.006
    L(wave.outputs[0], bump.inputs["Height"])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (0.36, 0.31, 0.22, 1)
    p.inputs["Metallic"].default_value = 0.55; p.inputs["Roughness"].default_value = 0.42
    L(bump.outputs[0], p.inputs["Normal"]); L(p.outputs[0], out.inputs[0])
    return m


def steel(name="Steel", color=(0.30, 0.29, 0.28), rough=0.45):
    return principled(name, color, metallic=1.0, rough=rough)


def rail_material():
    """Running band polished, sides rusty."""
    m = bpy.data.materials.get("Rail")
    if m:
        return m
    m = bpy.data.materials.new("Rail")
    nt, out = _nt(m)
    L = nt.links.new
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Normal"], sep.inputs[0])
    up = nt.nodes.new("ShaderNodeMapRange"); up.inputs[1].default_value = 0.93; up.inputs[2].default_value = 0.985
    L(sep.outputs["Z"], up.inputs[0])
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 40
    rust = nt.nodes.new("ShaderNodeMix"); rust.data_type = "RGBA"
    rust.inputs["A"].default_value = (0.18, 0.075, 0.035, 1); rust.inputs["B"].default_value = (0.11, 0.06, 0.035, 1)
    L(noise.outputs["Fac"], rust.inputs["Factor"])
    col = nt.nodes.new("ShaderNodeMix"); col.data_type = "RGBA"; col.inputs["B"].default_value = (0.62, 0.62, 0.62, 1)
    L(up.outputs[0], col.inputs["Factor"]); L(rust.outputs["Result"], col.inputs["A"])
    met = nt.nodes.new("ShaderNodeMapRange"); met.inputs[3].default_value = 0.2; met.inputs[4].default_value = 1.0
    L(up.outputs[0], met.inputs[0])
    rg = nt.nodes.new("ShaderNodeMapRange"); rg.inputs[3].default_value = 0.85; rg.inputs[4].default_value = 0.18
    L(up.outputs[0], rg.inputs[0])
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(col.outputs["Result"], p.inputs["Base Color"]); L(met.outputs[0], p.inputs["Metallic"]); L(rg.outputs[0], p.inputs["Roughness"])
    # the running band is polished lengthwise by the wheels: anisotropic highlight stretched along the rail
    # (sweep UVs: u across the profile, v along the chainage -> UV tangent rotated by 90 degrees)
    tg = nt.nodes.new("ShaderNodeTangent"); tg.direction_type = "UV_MAP"; tg.uv_map = "UVMap"
    L(tg.outputs[0], p.inputs["Tangent"])
    L(up.outputs[0], p.inputs["Anisotropic"])
    p.inputs["Anisotropic Rotation"].default_value = 0.25
    L(p.outputs[0], out.inputs[0])
    return m
