"""Per-frame physics log (pure python): speed, wheel rate, positions, camera speed and distance.

usage: python -m film.physics out.json
"""
import json
import math
import sys

import numpy as np

from . import alignment as A
from . import shots as SH
from . import specs as S


def frame_log():
    rows = []
    off = A.track_offset()
    for sh, f0, f1 in SH.timeline(SH.build_shots()):
        for f in range(f0, f1 + 1):
            t = (f - f0) / SH.FPS
            s = sh.s_head(t)
            nose = A.frame(np.array([s]), off)[0][0]
            r = dict(frame=f, shot=sh.name, t=round(t, 3), lens=sh.lens, s_head=round(s, 2),
                     speed_kmh=S.SERVICE_SPEED_KMH, speed_ms=round(S.SPEED, 3),
                     wheel_rps=round(S.WHEEL_REV_PER_S, 2), wheel_deg_frame=round(360 * S.WHEEL_REV_PER_S / SH.FPS, 1),
                     blur_m=round(S.SPEED * 0.5 / SH.FPS, 2),
                     curvature_per_km=round(float(A.curvature(s)) * 1000, 4), cant_mm=round(float(A.cant(s)) * 1000, 1))
            if sh.cam_fn is not None:
                c0 = np.asarray(sh.cam_fn(t)); c1 = np.asarray(sh.cam_fn(t + 1 / SH.FPS))
                r["cam_speed_ms"] = round(float(np.linalg.norm(c1 - c0) * SH.FPS), 2)
                # closest point of the train to the camera
                ss = s - np.linspace(0, S.TRAIN_LENGTH, 60)
                pts = A.frame(ss, off)[0] + np.array([0, 0, 2.0])
                r["cam_to_train_m"] = round(float(np.min(np.linalg.norm(pts - c0, axis=1))), 1)
            else:
                r["cam_speed_ms"] = round(S.SPEED, 2)
                r["cam_to_train_m"] = 0.0
            rows.append(r)
    return rows


if __name__ == "__main__":
    rows = frame_log()
    with open(sys.argv[1] if len(sys.argv) > 1 else "physics.json", "w") as fh:
        json.dump(rows, fh)
    print(f"[physics] {len(rows)} frames")
