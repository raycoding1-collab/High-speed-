"""Render single frames from the built scene.

usage: python -m tools.render_still scene.blend out_dir FRAME[,FRAME...] [--res 960x540] [--samples 24] [--novol]
"""
import argparse
import os
import sys
import time

import bpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blend"); ap.add_argument("outdir"); ap.add_argument("frames")
    ap.add_argument("--res", default="960x540"); ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--novol", action="store_true"); ap.add_argument("--noblur", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    import addon_utils
    addon_utils.enable("cycles", default_set=True)
    sc = bpy.context.scene
    w, h = map(int, a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.cycles.samples = a.samples
    if a.noblur:
        sc.render.use_motion_blur = False
    if a.novol:
        for o in bpy.data.objects:
            if o.name.startswith("Haze"):
                o.hide_render = True
    os.makedirs(a.outdir, exist_ok=True)
    for f in [int(x) for x in a.frames.split(",")]:
        sc.frame_set(f)
        cam = None
        for m in sorted(sc.timeline_markers, key=lambda m: m.frame):
            if m.frame <= f:
                cam = m.camera; name = m.name
        sc.camera = cam
        sc.render.filepath = os.path.join(a.outdir, f"{name}_f{f:04d}{a.tag}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[still] {name} frame {f}: {time.time() - t:.1f} s -> {sc.render.filepath}", flush=True)


if __name__ == "__main__":
    main()
