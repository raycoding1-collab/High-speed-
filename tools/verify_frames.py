"""Gate checks on rendered test frames (no Blender needed):

- exposure: % clipped (any channel >= 254), % crushed (luma <= 2), mean luma (display-referred 0-1);
- shadow neutrality: mean (G+B)/2 - R of the darkest 25 % of pixels, in 0-255 units; a strongly positive value is
  a teal cast in the shadows (the v1 compositor lift bug made this ~+14);
usage: python tools/verify_frames.py DIR [report.json]   (reads *.png / *.jpg)
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image


def stats(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    luma = a @ [0.2126, 0.7152, 0.0722]
    clip = float(np.mean(np.any(a >= 254, axis=-1)) * 100)
    crush = float(np.mean(luma <= 2) * 100)
    dark = luma <= np.percentile(luma, 25)
    d = a[dark]
    teal = float(((d[:, 1] + d[:, 2]) / 2 - d[:, 0]).mean())
    return dict(file=os.path.basename(path), clip_pct=round(clip, 2), crush_pct=round(crush, 2),
                mean_luma=round(float(luma.mean() / 255), 3), shadow_teal=round(teal, 1))


def main(d, report=None, teal_max=8.0, clip_max=1.0, crush_max=10.0):
    files = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    rows = [stats(f) for f in files]
    ok = True
    for r in rows:
        r["pass"] = r["shadow_teal"] < teal_max and r["clip_pct"] < clip_max and r["crush_pct"] < crush_max
        ok &= r["pass"]
        print(f"[frames] {r['file']}: clip {r['clip_pct']}%  crush {r['crush_pct']}%  mean {r['mean_luma']}  "
              f"shadow (G+B)/2-R {r['shadow_teal']:+.1f}  {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    if report:
        json.dump(dict(frames=rows, result="PASS" if ok else "FAIL"), open(report, "w"), indent=1)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None) else 1)
