"""Render the model from the solved camera of a reference photo and compose a comparison.

usage: python -m tools.match_ref <ref_id> <out.png> [--full] [--samples N]

World frame for matching: nose tip at origin, train runs towards +x, so a point at
design distance X from the tip lies at world x = -X; +y = train's left side.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REFDIR = os.environ.get("REFDIR", "/tmp/claude-0/research/ref")

# image point (px in the 1280-wide reference) <-> design point (X from tip, y, z)
REFS = {
    "ref08": dict(file="ref08.jpg", size=(1280, 853), pts=[
        ((68.3, 486.7), (0.0, 0.0, 1.00)),            # bill tip
        ((432.7, 398.3), (8.69, 1.66, 3.37)),         # cab door top front
        ((432.7, 478.3), (8.69, 1.67, 1.40)),         # cab door bottom front
        ((634.0, 395.0), (15.10, 1.68, 3.42)),        # pax door top front
        ((663.3, 488.3), (16.05, 1.68, 1.18)),        # pax door bottom rear
        ((942.0, 406.0), (28.55, 1.00, 3.80)),        # gangway ring top edge
        ((945.0, 527.5), (28.55, 1.60, 0.20)),        # gangway ring bottom
        ((70.0, 541.5), (0.0, 0.7175, 0.0)),          # near rail under tip
        ((1131.0, 534.0), (40.0, 0.7175, 0.0)),       # near rail at X=40
    ]),
    "ref03": dict(file="ref03.jpg", size=(1280, 853), pts=[
        ((1005.0, 462.5), (1.30, 0.0, 1.72)),         # gold logo centre
        ((558.0, 309.0), (8.69, -1.66, 3.37)),        # cab door top front
        ((553.0, 525.0), (8.69, -1.67, 1.40)),        # cab door bottom front
        ((390.0, 537.0), (15.10, -1.68, 1.18)),       # pax door bottom front
        ((203.0, 485.0), (28.55, -1.68, 2.00)),       # gangway ring at band bottom
        ((211.0, 600.0), (28.55, -1.60, 0.25)),       # gangway ring bottom
        ((670.0, 495.0), (5.10, -1.56, 1.55)),        # red wedge tip
        ((1110.0, 590.0), (0.0, 0.0, 1.00)),          # bill tip (approx)
    ]),
}


def solve_camera(ref):
    import cv2
    w, h = ref["size"]
    img = np.array([p[0] for p in ref["pts"]], dtype=np.float64)
    obj = np.array([(-X, y, z) for (X, y, z) in (p[1] for p in ref["pts"])], dtype=np.float64)
    K0 = np.array([[1500.0, 0, w / 2], [0, 1500.0, h / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32)], (w, h), K0, None, flags=flags)
    proj, _ = cv2.projectPoints(obj, rvecs[0], tvecs[0], K, dist)
    res = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    return K, rvecs[0], tvecs[0], err, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    ref = REFS[args.ref]
    K, rvec, tvec, err, res = solve_camera(ref)
    print(f"[match] rms {err:.2f}px  f={K[0,0]:.0f}px  per-point {np.round(res,1).tolist()}")

    import bpy
    from film import bl, train, specs as S, lookdev
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bl.set_cycles(sc, samples=args.samples)
    lookdev.simple_world(sc)
    cars = train.car_layout()
    coll = bl.collection("Train")
    c1 = train.build_end_car_shell("Car1", coll)
    c1.location = (-(cars[0]["centre"]), 0, 0)
    c2 = train.build_mid_car_shell("Car2", coll)
    c2.location = (-(cars[1]["centre"]), 0, 0)
    lookdev.apply_temp_materials([c1, c2])
    sc.render.film_transparent = True
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam); sc.camera = cam
    w, h = ref["size"]
    bl.camera_from_opencv(cam, rvec, tvec, K[0, 0], w, h)
    sc.render.resolution_x = int(w * args.scale); sc.render.resolution_y = int(h * args.scale)
    tmp = args.out + ".render.png"
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)

    from PIL import Image, ImageFilter, ImageChops
    a = Image.open(os.path.join(REFDIR, ref["file"])).convert("RGB").resize((w, h))
    rgba = Image.open(tmp).convert("RGBA").resize((w, h))
    alpha = rgba.split()[3].point(lambda v: 255 if v > 127 else 0)
    edge = ImageChops.subtract(alpha.filter(ImageFilter.MaxFilter(3)), alpha.filter(ImageFilter.MinFilter(3)))
    outl = a.copy(); red = Image.new("RGB", (w, h), (255, 0, 0)); outl.paste(red, (0, 0), edge)
    comp = a.copy(); comp.paste(rgba.convert("RGB"), (0, 0), rgba.split()[3])
    sheet = Image.new("RGB", (w, h * 3))
    sheet.paste(a, (0, 0)); sheet.paste(comp, (0, h)); sheet.paste(outl, (0, 2 * h))
    sheet.save(args.out)
    print("[match] wrote", args.out)


if __name__ == "__main__":
    main()
