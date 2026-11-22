"""Render frames of the packed scene on Modal GPUs (Cycles OptiX/CUDA), resumable.

The packed .blend is first copied into the Modal volume "cr450-scene" (see .github/workflows/render-modal.yml).
Each GPU container opens the scene once and then renders the frames it is handed, so the ~40 s scene load is
paid once per container, not once per frame.

usage (GitHub Actions, with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment):
    modal run tools/modal_render.py --blend scene-123.blend --frames 1-1560 --out frames [--res 1920x1080] [--samples 48]
"""
import json
import os
import time

import modal

GPU = os.environ.get("MODAL_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "10"))
CPU = float(os.environ.get("MODAL_CPU", "8"))          # scene sync / geometry nodes / BVH build run on the CPU
MEMORY = int(os.environ.get("MODAL_MEMORY", "16384"))

app = modal.App("cr450-render")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("xorg", "libxkbcommon0", "libsm6", "libgl1")       # Blender's X11/GL runtime libraries
    .uv_pip_install("bpy==5.3.0", "numpy==2.5.5")
)
vol = modal.Volume.from_name("cr450-scene", create_if_missing=True)


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@app.cls(gpu=GPU, image=image, volumes={"/scene": vol}, max_containers=MAX_CONTAINERS, timeout=3600,
         cpu=CPU, memory=MEMORY)
class Renderer:
    blend: str = modal.parameter()
    res: str = modal.parameter(default="")
    samples: int = modal.parameter(default=0)

    @modal.enter()
    def load(self):
        import bpy                      # must come first: it puts Blender's own modules (addon_utils) on sys.path
        import addon_utils
        t = time.time()
        bpy.ops.wm.open_mainfile(filepath=f"/scene/{self.blend}")
        addon_utils.enable("cycles", default_set=True)
        sc = bpy.context.scene
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    break
            except TypeError:
                continue
        for d in prefs.devices:
            d.use = d.type == prefs.compute_device_type
        sc.cycles.device = "GPU"
        self.devices = [f"{d.name} ({d.type})" for d in prefs.devices if d.use]
        if not self.devices:
            raise RuntimeError("no GPU visible to Cycles")
        if self.res:
            w, h = map(int, self.res.split("x"))
            sc.render.resolution_x, sc.render.resolution_y = w, h
        sc.render.resolution_percentage = 100
        if self.samples:
            sc.cycles.samples = self.samples
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.render.image_settings.compression = 15
        sc.render.use_persistent_data = True
        self.markers = sorted(sc.timeline_markers, key=lambda m: m.frame)
        print(f"[modal] scene loaded in {time.time() - t:.0f} s on {self.devices}", flush=True)

    @modal.method()
    def render(self, frame: int):
        import bpy
        sc = bpy.context.scene
        cam = None
        for m in self.markers:
            if m.frame <= frame:
                cam = m.camera
        sc.camera = cam
        sc.frame_set(frame)
        path = f"/tmp/frame_{frame:05d}.png"
        sc.render.filepath = path
        t = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t
        with open(path, "rb") as fh:
            data = fh.read()
        os.remove(path)
        return frame, data, dt, (cam.name if cam else "-"), self.devices[0]


@app.local_entrypoint()
def main(blend: str, frames: str, out: str, res: str = "", samples: int = 0):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render_frames import png_complete
    os.makedirs(out, exist_ok=True)
    wanted = parse_frames(frames)
    todo = [f for f in wanted if not png_complete(os.path.join(out, f"frame_{f:05d}.png"))]
    print(f"[modal] {len(wanted)} frames wanted, {len(wanted) - len(todo)} already done, rendering {len(todo)} "
          f"on up to {MAX_CONTAINERS} x {GPU}", flush=True)
    t0 = time.time()
    timings, failed = [], []
    r = Renderer(blend=blend, res=res, samples=samples)
    for item in r.render.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(item, BaseException):
            failed.append(repr(item)[:300])
            print(f"[modal] FAILED: {failed[-1]}", flush=True)
            continue
        f, data, dt, cam, dev = item
        tmp = os.path.join(out, f".tmp_{f:05d}.png")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, os.path.join(out, f"frame_{f:05d}.png"))
        timings.append(dict(frame=f, seconds=round(dt, 2), camera=cam, device=dev))
        print(f"[modal] frame {f} ({cam}) {dt:.1f} s on {dev}  [{len(timings)}/{len(todo)}]", flush=True)
        if len(timings) % 25 == 0:                       # progress line with a running cost estimate
            el = time.time() - t0
            print(f"[modal] progress {len(timings)}/{len(todo)} after {el/60:.1f} min, "
                  f"ETA {el / len(timings) * (len(todo) - len(timings)) / 60:.0f} min", flush=True)
    wall = time.time() - t0
    with open(os.path.join(out, "timing_modal.json"), "w") as fh:
        json.dump(dict(gpu=GPU, containers=MAX_CONTAINERS, wall_s=round(wall, 1), frames=timings, failed=failed), fh, indent=1)
    done = sum(png_complete(os.path.join(out, f"frame_{f:05d}.png")) for f in wanted)
    if timings:
        s = [t["seconds"] for t in timings]
        print(f"[modal] rendered {len(s)} frames in {wall / 60:.1f} min wall, mean {sum(s) / len(s):.1f} s/frame "
              f"(GPU render time only), max {max(s):.1f} s", flush=True)
    print(f"[modal] {done}/{len(wanted)} frames complete", flush=True)
    if done != len(wanted):
        raise SystemExit(3)
