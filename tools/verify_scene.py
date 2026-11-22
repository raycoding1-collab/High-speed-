"""Gate checks on a built scene (run with Blender's Python / bpy):

- every scattered / instanced asset is evaluated with finite transforms, counted per scatter object, at the middle
  frame of every shot (a NaN in a geometry-nodes effect once made all vegetation invisible without any error);
- every image is packed or present on disk (no pink missing textures on the render machines);
- exposure key per shot, frame range and camera markers.

usage: python tools/verify_scene.py scene.blend [report.json]
"""
import json
import math
import os
import sys

import bpy


def main(blend, report=None):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    marks = sorted(sc.timeline_markers, key=lambda m: m.frame)
    shots = []
    for i, m in enumerate(marks):
        f1 = marks[i + 1].frame - 1 if i + 1 < len(marks) else sc.frame_end
        shots.append((m.name, m.frame, f1))
    out = dict(blend=os.path.basename(blend), frames=[sc.frame_start, sc.frame_end], shots=[], images={})
    bad_total = 0
    for name, f0, f1 in shots:
        f = (f0 + f1) // 2
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        cnt, bad = {}, {}
        for inst in dg.object_instances:
            if not inst.is_instance or inst.parent is None:
                continue
            nm = inst.parent.name
            M = inst.matrix_world
            ok = all(math.isfinite(M[r][c]) for r in range(3) for c in range(4))
            cnt[nm] = cnt.get(nm, 0) + 1
            if not ok:
                bad[nm] = bad.get(nm, 0) + 1
        bad_total += sum(bad.values())
        row = dict(shot=name, frame=f, camera=sc.camera.name if sc.camera else None,
                   exposure=round(sc.view_settings.exposure, 2), instances=sum(cnt.values()),
                   invalid=sum(bad.values()), per_object={k: cnt[k] for k in sorted(cnt)})
        out["shots"].append(row)
        print(f"[verify] {name} f{f}: {row['instances']} instances, {row['invalid']} invalid, exposure {row['exposure']:+.1f}", flush=True)
    missing = []
    for im in bpy.data.images:
        if im.source != "FILE":
            continue
        packed = im.packed_file is not None
        exists = packed or os.path.exists(bpy.path.abspath(im.filepath))
        if not exists:
            missing.append(im.name)
    out["images"] = dict(total=len([i for i in bpy.data.images if i.source == "FILE"]),
                         packed=len([i for i in bpy.data.images if i.packed_file is not None]), missing=missing)
    houses = [o for o in bpy.data.objects if o.data is not None and getattr(o.data, "name", "").startswith("House_")]
    out["houses"] = len(houses)
    out["result"] = "PASS" if bad_total == 0 and not missing else "FAIL"
    print(f"[verify] images {out['images']}, houses {len(houses)} -> {out['result']}", flush=True)
    if report:
        json.dump(out, open(report, "w"), indent=1)
    return out["result"] == "PASS"


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ok = main(args[0], args[1] if len(args) > 1 else None)
    sys.exit(0 if ok else 1)
