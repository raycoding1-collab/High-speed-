"""Neutral look-development helpers (temporary materials, simple world, ground)."""
import math

import bpy


def simple_world(scene, sun_elev=35.0, sun_rot=200.0, strength=0.25):
    w = bpy.data.worlds.new("LookdevWorld")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(sun_elev)
    sky.sun_rotation = math.radians(sun_rot)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w


def apply_temp_materials(objs):
    m = bpy.data.materials.get("TempSilver")
    if m is None:
        m = bpy.data.materials.new("TempSilver")
        m.use_nodes = True
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (0.62, 0.63, 0.65, 1)
        p.inputs["Metallic"].default_value = 0.9
        p.inputs["Roughness"].default_value = 0.28
        p.inputs["Coat Weight"].default_value = 1.0
        p.inputs["Coat Roughness"].default_value = 0.03
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(m)


def ground(scene, size=400, color=(0.18, 0.17, 0.16)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, -0.75))
    g = bpy.context.active_object
    g.name = "Ground"
    m = bpy.data.materials.new("Ground")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    g.data.materials.append(m)
    return g
