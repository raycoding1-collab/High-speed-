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
    cy.adaptive_threshold = 0.02
    cy.use_denoising = denoise
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 6
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 4
    cy.transmission_bounces = 6
    cy.transparent_max_bounces = 12
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
