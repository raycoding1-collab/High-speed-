"""Small bpy helpers shared by all builders."""
import math

import bpy
import numpy as np


def new_mesh_object(name, verts, faces, collection=None, uvs=None, smooth=True, sharp_edges=None, creases=None):
    """Create an object from numpy verts (N,3) and a list/array of faces.

    uvs: optional per-loop UV array (L,2) matching the flattened face loop order.
    sharp_edges / creases: optional iterables of (v0, v1) vertex index pairs.
    """
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, dtype=np.float64)
    faces = [list(map(int, f)) for f in faces]
    me.from_pydata(verts.tolist(), [], faces)
    me.validate(clean_customdata=False)
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if sharp_edges or creases:
        key_to_idx = {tuple(sorted(e.vertices)): i for i, e in enumerate(me.edges)}
        if sharp_edges:
            attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
            vals = np.zeros(len(me.edges), dtype=bool)
            for a, b in sharp_edges:
                i = key_to_idx.get(tuple(sorted((int(a), int(b)))))
                if i is not None:
                    vals[i] = True
            attr.data.foreach_set("value", vals)
        if creases:
            attr = me.attributes.get("crease_edge") or me.attributes.new("crease_edge", "FLOAT", "EDGE")
            vals = np.zeros(len(me.edges), dtype=np.float32)
            for a, b in creases:
                i = key_to_idx.get(tuple(sorted((int(a), int(b)))))
                if i is not None:
                    vals[i] = 1.0
            attr.data.foreach_set("value", vals)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def grid_faces(n_rings, ring_len, closed=True):
    """Quad faces between consecutive rings of equal length."""
    faces = []
    m = ring_len if closed else ring_len - 1
    for i in range(n_rings - 1):
        a = i * ring_len; b = (i + 1) * ring_len
        for j in range(m):
            j2 = (j + 1) % ring_len
            faces.append((a + j, a + j2, b + j2, b + j))
    return faces


def collection(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def set_cycles(scene, samples=64, denoise=True):
    import addon_utils
    addon_utils.enable("cycles", default_set=True)
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.03
    cy.adaptive_min_samples = 12
    scene.render.use_persistent_data = True
    cy.use_denoising = denoise
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 4
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 4
    cy.transparent_max_bounces = 10
    cy.ao_bounces_render = 2
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 8.0
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"


def look_at(ob, target, roll=0.0):
    from mathutils import Vector
    d = Vector(target) - ob.location
    q = d.to_track_quat("-Z", "Y")
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q
    if roll:
        from mathutils import Quaternion
        ob.rotation_quaternion = ob.rotation_quaternion @ Quaternion((0, 0, 1), roll)


def camera_from_opencv(cam_ob, rvec, tvec, fx, width, height, sensor_w=36.0):
    """Place a Blender camera from OpenCV extrinsics (world->camera) and focal in px."""
    import cv2
    from mathutils import Matrix
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    t = np.asarray(tvec, float).reshape(3)
    Rw = R.T; C = -R.T @ t                    # camera centre in world
    flip = np.diag([1, -1, -1])               # opencv cam (x r, y down, z fwd) -> blender cam (x r, y up, -z fwd)
    Rb = Rw @ flip
    M = Matrix(((Rb[0, 0], Rb[0, 1], Rb[0, 2], C[0]), (Rb[1, 0], Rb[1, 1], Rb[1, 2], C[1]),
                (Rb[2, 0], Rb[2, 1], Rb[2, 2], C[2]), (0, 0, 0, 1)))
    cam_ob.matrix_world = M
    cam = cam_ob.data
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = sensor_w
    cam.lens = fx * sensor_w / width


def instancer(name, src, positions, rotations=None, scales=None, coll=None, extra=None):
    """Point cloud with per-point rotation (euler XYZ) / scale, instancing `src` via geometry nodes."""
    positions = np.asarray(positions, np.float32)
    n = len(positions)
    me = bpy.data.meshes.new(name + "_pts")
    me.vertices.add(n)
    me.vertices.foreach_set("co", positions.ravel())
    rot = np.zeros((n, 3), np.float32) if rotations is None else np.asarray(rotations, np.float32)
    scl = np.ones((n, 3), np.float32) if scales is None else np.asarray(scales, np.float32).reshape(n, -1) * np.ones((1, 3), np.float32)
    a = me.attributes.new("rot", "FLOAT_VECTOR", "POINT"); a.data.foreach_set("vector", rot.ravel())
    b = me.attributes.new("scl", "FLOAT_VECTOR", "POINT"); b.data.foreach_set("vector", scl.ravel())
    if extra:
        for k, v in extra.items():
            c = me.attributes.new(k, "FLOAT", "POINT"); c.data.foreach_set("value", np.asarray(v, np.float32))
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ng = _instancer_tree()
    md = ob.modifiers.new("inst", "NODES"); md.node_group = ng
    _set_modifier_input(md, "Object", src)
    return ob


def modifier_input(md, name):
    """Return the RNA struct holding a geometry-nodes modifier input (Blender 5.x: md.properties.inputs.<id>)."""
    for item in md.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == name:
            return getattr(md.properties.inputs, item.identifier)
    raise KeyError(name)


def _set_modifier_input(md, name, value):
    modifier_input(md, name).value = value


def _instancer_tree():
    ng = bpy.data.node_groups.get("PointInstancer")
    if ng:
        return ng
    ng = bpy.data.node_groups.new("PointInstancer", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Object", in_out="INPUT", socket_type="NodeSocketObject")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; L = ng.links.new
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    oi = N.new("GeometryNodeObjectInfo"); oi.transform_space = "ORIGINAL"; oi.inputs["As Instance"].default_value = True
    ra = N.new("GeometryNodeInputNamedAttribute"); ra.data_type = "FLOAT_VECTOR"; ra.inputs["Name"].default_value = "rot"
    sa = N.new("GeometryNodeInputNamedAttribute"); sa.data_type = "FLOAT_VECTOR"; sa.inputs["Name"].default_value = "scl"
    e2r = N.new("FunctionNodeEulerToRotation")
    iop = N.new("GeometryNodeInstanceOnPoints")
    L(gi.outputs["Object"], oi.inputs["Object"])
    L(gi.outputs["Geometry"], iop.inputs["Points"])
    L(oi.outputs["Geometry"], iop.inputs["Instance"])
    L(ra.outputs["Attribute"], e2r.inputs["Euler"]); L(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    L(sa.outputs["Attribute"], iop.inputs["Scale"])
    L(iop.outputs["Instances"], go.inputs["Geometry"])
    return ng


def matrix_to_euler(T, N, U):
    """Rows: local x=T, y=N, z=U (world vectors) -> XYZ euler (n,3)."""
    from mathutils import Matrix
    T = np.atleast_2d(T); N = np.atleast_2d(N); U = np.atleast_2d(U)
    out = np.zeros((len(T), 3))
    for i in range(len(T)):
        m = Matrix(((T[i, 0], N[i, 0], U[i, 0]), (T[i, 1], N[i, 1], U[i, 1]), (T[i, 2], N[i, 2], U[i, 2])))
        out[i] = m.to_euler("XYZ")
    return out


def set_keys(id_owner, data_path, index, frames, values, interp="LINEAR"):
    """Bulk-create one F-curve (Blender 5 layered actions) with linear keys."""
    ad = id_owner.animation_data or id_owner.animation_data_create()
    act = ad.action
    if act is None:
        act = bpy.data.actions.new(id_owner.name + "_act")
        ad.action = act
    slot = ad.action_slot
    if slot is None:
        slot = act.slots.new(id_type=id_owner.id_type if hasattr(id_owner, "id_type") else "OBJECT", name=id_owner.name)
        ad.action_slot = slot
    layer = act.layers[0] if len(act.layers) else act.layers.new("Layer")
    strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type="KEYFRAME")
    cb = strip.channelbag(slot, ensure=True)
    fc = cb.fcurves.find(data_path, index=index) or cb.fcurves.new(data_path, index=index)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n, np.float32); co[0::2] = frames; co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    ii = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}[interp]
    fc.keyframe_points.foreach_set("interpolation", np.full(n, ii, np.int32))
    fc.update()
    return fc


def set_matrix_keys(ob, frames, mats):
    """mats: (n,4,4) world matrices -> location + XYZ euler keys (euler made continuous)."""
    from mathutils import Matrix
    n = len(frames)
    loc = np.zeros((n, 3)); rot = np.zeros((n, 3))
    prev = None
    for i, M in enumerate(mats):
        m = Matrix(M.tolist())
        loc[i] = m.translation
        e = m.to_euler("XYZ", prev) if prev is not None else m.to_euler("XYZ")
        rot[i] = e; prev = e
    ob.rotation_mode = "XYZ"
    for k in range(3):
        set_keys(ob, "location", k, frames, loc[:, k])
        set_keys(ob, "rotation_euler", k, frames, rot[:, k])
