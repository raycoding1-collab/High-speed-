"""Render an interleaved subset of frames from the built scene (one GitHub Actions matrix job).

Job k of N renders frames  start+k, start+k+N, start+k+2N, ...  so every job gets a similar mix of
easy and heavy shots.  Resumable: frames that already exist as complete PNGs are skipped.  Stops cleanly
before `--deadline-min` so partial results can be uploaded and the job re-run later.

usage: python tools/render_frames.py --blend scene.blend --out frames --job K --jobs N
       [--frames 1-1560 | --frames 100,1000] [--deadline-min 330] [--timing timing.json]
"""
import argparse
import json
import os
import sys
import time


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def png_complete(path):
    """True if the PNG exists, ends with an IEND chunk and decodes."""
    try:
        if os.path.getsize(path) < 1000:
            return False
        with open(path, "rb") as f:
            f.seek(-12, 2)
            if f.read(12)[4:8] != b"IEND":
                return False
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--job", type=int, default=0); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--frames", default=None); ap.add_argument("--deadline-min", type=float, default=330)
    ap.add_argument("--timing", default=None); ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--res", default=None)
    a = ap.parse_args()
    t_start = time.time()
    import bpy
    import addon_utils
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    addon_utils.enable("cycles", default_set=True)
    sc = bpy.context.scene
    sc.render.use_persistent_data = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 15
    if a.samples:
        sc.cycles.samples = a.samples
    if a.res:
        w, h = map(int, a.res.split("x")); sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    frames = parse_frames(a.frames) if a.frames else list(range(sc.frame_start, sc.frame_end + 1))
    mine = frames[a.job::a.jobs]
    os.makedirs(a.out, exist_ok=True)
    markers = sorted(sc.timeline_markers, key=lambda m: m.frame)
    timings = []
    todo = [f for f in mine if not png_complete(os.path.join(a.out, f"frame_{f:05d}.png"))]
    print(f"[render] job {a.job}/{a.jobs}: {len(mine)} frames assigned, {len(mine) - len(todo)} already done, "
          f"{sc.render.resolution_x}x{sc.render.resolution_y} @ {sc.cycles.samples} spp", flush=True)
    for f in todo:
        elapsed = (time.time() - t_start) / 60
        est = (max(timings) if timings else 0) / 60
        if elapsed + 1.3 * est > a.deadline_min:
            print(f"[render] deadline reached after {elapsed:.1f} min; stopping cleanly", flush=True)
            break
        cam = None
        for m in markers:
            if m.frame <= f:
                cam = m.camera
        sc.camera = cam
        sc.frame_set(f)
        dst = os.path.join(a.out, f"frame_{f:05d}.png")
        tmp = os.path.join(a.out, f".tmp_{f:05d}.png")
        sc.render.filepath = tmp
        t = time.time()
        bpy.ops.render.render(write_still=True)
        os.replace(tmp, dst)
        dt = time.time() - t
        timings.append(dt)
        print(f"[render] frame {f} ({cam.name if cam else '-'}) {dt:.1f} s", flush=True)
    done = sum(png_complete(os.path.join(a.out, f"frame_{f:05d}.png")) for f in mine)
    if a.timing:
        with open(a.timing, "w") as fh:
            json.dump(dict(job=a.job, jobs=a.jobs, frames=todo[:len(timings)], seconds=timings,
                           res=[sc.resolution_x if hasattr(sc, "resolution_x") else sc.render.resolution_x, sc.render.resolution_y],
                           samples=sc.cycles.samples, total_min=(time.time() - t_start) / 60), fh, indent=1)
    print(f"[render] job {a.job}: {done}/{len(mine)} frames complete, {(time.time() - t_start) / 60:.1f} min", flush=True)
    sys.exit(0 if done == len(mine) else 3)


if __name__ == "__main__":
    main()
