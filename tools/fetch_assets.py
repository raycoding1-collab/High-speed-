"""Download the pinned CC0 Poly Haven assets used by the scene (verified by MD5).

usage: python tools/fetch_assets.py <assets_dir>
The same script runs locally and in the GitHub Actions build job, so every machine
builds the scene from byte-identical files.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "train-film-build/1.0"}

HDRIS = [("kloppenheim_06_puresky", "8k"), ("evening_road_01_puresky", "4k")]
TEXTURES = [  # (id, res, maps)
    ("concrete_wall_008", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("precast_concrete_wall", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("rock_face_03", "2k", ["Diffuse", "Rough", "nor_gl", "Displacement"]),
    ("marble_cliff_02", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("forest_ground_04", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("sparse_grass", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("brown_mud_02", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("gravel_floor_02", "2k", ["Diffuse", "Rough", "nor_gl"]),
    ("aerial_grass_rock", "2k", ["Diffuse", "Rough", "nor_gl"]),
]
MODELS = [("island_tree_01", "1k"), ("island_tree_02", "1k"), ("island_tree_03", "1k"), ("tree_small_02", "1k"),
          ("shrub_01", "1k"), ("shrub_02", "1k"), ("shrub_04", "1k"), ("fern_02", "1k"), ("grass_medium_01", "1k"),
          ("grass_medium_02", "1k"), ("weed_plant_02", "1k"), ("boulder_01", "1k")]


def get_json(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"failed {url}")


def fetch(url, dst, md5=None, size=None):
    if os.path.exists(dst) and (md5 is None or _md5(dst) == md5):
        return dst, "cached"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(5):
        try:
            tmp = dst + ".part"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if md5 and _md5(tmp) != md5:
                raise IOError("md5 mismatch")
            os.replace(tmp, dst)
            return dst, "ok"
        except Exception as e:  # noqa
            err = e
            time.sleep(3 + 5 * k)
    raise RuntimeError(f"download failed {url}: {err}")


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jobs(root):
    out = []
    for aid, res in HDRIS:
        d = get_json(API + aid)["hdri"][res]["hdr"]
        out.append((d["url"], os.path.join(root, "hdri", f"{aid}_{res}.hdr"), d.get("md5")))
    for aid, res, maps in TEXTURES:
        d = get_json(API + aid)
        for m in maps:
            fmt = "png" if m in ("nor_gl", "Displacement") and "png" in d[m][res] else "jpg"
            e = d[m][res][fmt]
            out.append((e["url"], os.path.join(root, "tex", aid, os.path.basename(e["url"])), e.get("md5")))
    for aid, res in MODELS:
        d = get_json(API + aid)["blend"][res]["blend"]
        base = os.path.join(root, "models", aid)
        out.append((d["url"], os.path.join(base, os.path.basename(d["url"])), d.get("md5")))
        for rel, e in d.get("include", {}).items():
            out.append((e["url"], os.path.join(base, rel), e.get("md5")))
    return out


def main(root):
    js = jobs(root)
    print(f"[assets] {len(js)} files")
    with cf.ThreadPoolExecutor(4) as ex:
        for dst, st in ex.map(lambda j: fetch(*j), js):
            if st != "cached":
                print("[assets]", st, os.path.relpath(dst, root))
    print("[assets] done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets")
