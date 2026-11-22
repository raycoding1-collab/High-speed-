"""Compositing: aerial-perspective haze from the Mist pass, bloom and a gentle golden-hour grade.

Haze: T = exp(-d / V) with visibility V; the sky (mist == 1) is left untouched because the HDRI already
contains its own horizon haze.  This replaces a kilometre-tall scattering volume (the valley-floor mist
layer stays a real volume), which halves the path-tracing cost of wide shots.
"""
import bpy

MIST_DEPTH = 20000.0


def setup(scene, visibility=7500.0, haze=(0.78, 0.70, 0.60), haze_gain=0.9, grade=True, haze_on=True):
    scene.render.compositor_device = "CPU"           # no GPU on the render machines
    scene.render.compositor_denoise_device = "CPU" if hasattr(scene.render, "compositor_denoise_device") else None
    scene.render.use_compositing = True
    vl = scene.view_layers[0]
    vl.use_pass_mist = True
    w = scene.world
    w.mist_settings.start = 0.0
    w.mist_settings.depth = MIST_DEPTH
    w.mist_settings.falloff = "LINEAR"
    ng = bpy.data.node_groups.new("Post", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N = ng.nodes; L = ng.links.new
    rl = N.new("CompositorNodeRLayers"); rl.scene = scene
    out = N.new("NodeGroupOutput")

    def math(op, a, b=None):
        n = N.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]
    # the Mist pass is not denoised by Cycles: run it through OIDN so the haze does not inherit sample noise
    cc = N.new("CompositorNodeCombineColor")
    for i in range(3):
        L(rl.outputs["Mist"], cc.inputs[i])
    dn = N.new("CompositorNodeDenoise")
    L(cc.outputs[0], dn.inputs["Image"])
    sep = N.new("CompositorNodeSeparateColor"); L(dn.outputs[0], sep.inputs[0])
    mist = sep.outputs[0]
    mist_raw = rl.outputs["Mist"]
    dist_k = math("MULTIPLY", mist, -MIST_DEPTH / visibility)
    fog = math("SUBTRACT", 1.0, math("EXPONENT", dist_k))
    sky = math("GREATER_THAN", mist_raw, 0.9995)
    amount = math("MULTIPLY", math("MULTIPLY", fog, math("SUBTRACT", 1.0, sky)), haze_gain)
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MIX"
    mix.inputs["B"].default_value = (*haze, 1.0)
    L(amount, mix.inputs["Factor"]); L(rl.outputs["Image"], mix.inputs["A"])
    img = mix.outputs["Result"] if haze_on else rl.outputs["Image"]
    if grade:
        gl = N.new("CompositorNodeGlare")
        try:
            gl.inputs["Type"].default_value = "Fog Glow"
        except Exception:
            try:
                gl.glare_type = "FOG_GLOW"
            except Exception:
                pass
        for nm, v in (("Threshold", 2.0), ("Strength", 0.12), ("Size", 0.5)):
            if nm in gl.inputs:
                gl.inputs[nm].default_value = v
        L(img, gl.inputs["Image"]); img = gl.outputs["Image"]
        cb = N.new("CompositorNodeColorBalance")
        try:
            cb.correction_method = "LIFT_GAMMA_GAIN"
        except Exception:
            pass
        # no lift: the compositor works on scene-linear values, where shadows are ~0.01-0.03, so a +-0.02 lift
        # (meant for display-referred images) turned every shaded area deep teal. Keep only a gentle warm gain.
        for nm, v in (("Lift", (1.0, 1.0, 1.0, 1)), ("Gain", (1.03, 1.0, 0.97, 1))):
            for s in cb.inputs:
                if s.name == nm and s.type == "RGBA":
                    s.default_value = v
        L(img, cb.inputs["Image"]); img = cb.outputs["Image"]
        # real-lens lateral chromatic aberration: radial dispersion, zero at the centre, ~1 px at the frame edges
        ld = N.new("CompositorNodeLensdist")
        for nm, v in (("Distortion", 0.0), ("Dispersion", 0.003)):
            if nm in ld.inputs:
                ld.inputs[nm].default_value = v
        L(img, ld.inputs["Image"]); img = ld.outputs["Image"]
    L(img, out.inputs["Image"])
    return ng
