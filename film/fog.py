"""Analytic atmospheric fog evaluated inside every surface shader (noise-free, nearly free to render).

Extinction  sigma(h) = s_mist * exp(-(h - h0) / H_mist) + s_haze * exp(-(h - h0) / H_haze)
Along the camera ray from camera height hc to surface height hp over distance d, each exponential layer has the
closed-form optical depth
      tau = s * H * d * (exp(-(hc-h0)/H) - exp(-(hp-h0)/H)) / (hp - hc)     (-> s * d * exp(-(h-h0)/H) when hp ~ hc)
Transmittance T = exp(-tau); the surface colour becomes  T * surface + (1 - T) * L_in  where the in-scattered
radiance L_in is a sky-ambient term plus a forward-scattering (Henyey-Greenstein) sun term, so the mist glows
warm towards the low sun exactly as in golden-hour valleys.  Camera height is recovered per shading point as
P + I * d (I = Incoming, d = Camera Data view distance).
"""
import math

import bpy

GROUP = "AtmosphericFog"


def _group(sun_dir, s_mist, h_mist, s_haze, h_haze, h0, amb, sun_col, g):
    ng = bpy.data.node_groups.get(GROUP)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Shader", in_out="INPUT", socket_type="NodeSocketShader")
    ng.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    N = ng.nodes; L = ng.links.new
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    geo = N.new("ShaderNodeNewGeometry"); cam = N.new("ShaderNodeCameraData")

    def m(op, a, b=None):
        n = N.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]
    d = cam.outputs["View Distance"]
    sp = N.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Position"], sp.inputs[0])
    si = N.new("ShaderNodeSeparateXYZ"); L(geo.outputs["Incoming"], si.inputs[0])
    hp = m("SUBTRACT", sp.outputs["Z"], h0)
    hc = m("ADD", hp, m("MULTIPLY", si.outputs["Z"], d))
    dh = m("SUBTRACT", hp, hc)

    def layer_tau(s, H):
        ec = m("EXPONENT", m("DIVIDE", hc, -H))
        ep = m("EXPONENT", m("DIVIDE", hp, -H))
        num = m("MULTIPLY", m("MULTIPLY", s * H, d), m("SUBTRACT", ec, ep))
        safe = m("MAXIMUM", m("ABSOLUTE", dh), 0.05)
        sgn = m("SIGN", dh)
        sgn = m("ADD", sgn, m("MULTIPLY", m("COMPARE", sgn, 0.0), 1.0))
        exact = m("DIVIDE", num, m("MULTIPLY", safe, sgn))
        mid = m("MULTIPLY", m("MULTIPLY", s, d), m("EXPONENT", m("DIVIDE", m("MULTIPLY", m("ADD", hc, hp), 0.5), -H)))
        small = m("LESS_THAN", m("ABSOLUTE", dh), 0.05)
        return m("ADD", m("MULTIPLY", small, mid), m("MULTIPLY", m("SUBTRACT", 1.0, small), exact))
    tau = m("ADD", layer_tau(s_mist, h_mist), layer_tau(s_haze, h_haze))
    T = m("EXPONENT", m("MULTIPLY", tau, -1.0))
    fogf = m("SUBTRACT", 1.0, T)
    # in-scattered radiance: ambient + HG(g) sun phase (cos between view ray and sun)
    vdir = N.new("ShaderNodeVectorMath"); vdir.operation = "SCALE"; L(geo.outputs["Incoming"], vdir.inputs[0]); vdir.inputs["Scale"].default_value = -1.0
    dot = N.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"; L(vdir.outputs[0], dot.inputs[0]); dot.inputs[1].default_value = tuple(sun_dir)
    cosv = dot.outputs["Value"]
    denom = m("POWER", m("SUBTRACT", 1 + g * g, m("MULTIPLY", cosv, 2 * g)), 1.5)
    phase = m("DIVIDE", (1 - g * g) / (4 * math.pi), denom)
    sunterm = m("MULTIPLY", phase, 4 * math.pi * 0.55)
    col = N.new("ShaderNodeMix"); col.data_type = "RGBA"; col.blend_type = "ADD"
    col.inputs["A"].default_value = (*amb, 1)
    sc = N.new("ShaderNodeMix"); sc.data_type = "RGBA"; sc.blend_type = "MULTIPLY"
    sc.inputs["A"].default_value = (*sun_col, 1); sc.inputs["Factor"].default_value = 1.0
    cmb = N.new("ShaderNodeCombineColor"); L(sunterm, cmb.inputs[0]); L(sunterm, cmb.inputs[1]); L(sunterm, cmb.inputs[2])
    L(cmb.outputs[0], sc.inputs["B"])
    col.inputs["Factor"].default_value = 1.0
    L(sc.outputs["Result"], col.inputs["B"])
    em = N.new("ShaderNodeEmission"); L(col.outputs["Result"], em.inputs["Color"]); em.inputs["Strength"].default_value = 1.0
    mix = N.new("ShaderNodeMixShader"); L(fogf, mix.inputs[0]); L(gi.outputs["Shader"], mix.inputs[1]); L(em.outputs[0], mix.inputs[2])
    L(mix.outputs[0], go.inputs["Shader"])
    return ng


def apply(scene, sun_dir, s_mist=0.0016, h_mist=11.0, s_haze=0.00012, h_haze=900.0, h0=1.5,
          amb=(0.62, 0.62, 0.64), sun_col=(1.0, 0.72, 0.46), g=0.62, skip=("Mist",)):
    """Wrap the surface output of every material with the fog group (idempotent)."""
    ng = _group(sun_dir, s_mist, h_mist, s_haze, h_haze, h0, amb, sun_col, g)
    n = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.name.startswith(skip) or mat.get("fogged"):
            continue
        nt = mat.node_tree
        outs = [o for o in nt.nodes if o.type == "OUTPUT_MATERIAL"]
        for o in outs:
            if not o.inputs["Surface"].is_linked:
                continue
            link = o.inputs["Surface"].links[0]
            src = link.from_socket
            g_node = nt.nodes.new("ShaderNodeGroup"); g_node.node_tree = ng
            nt.links.remove(link)
            nt.links.new(src, g_node.inputs[0]); nt.links.new(g_node.outputs[0], o.inputs["Surface"])
            n += 1
        mat["fogged"] = 1
    return n
