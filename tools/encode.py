"""Collect, validate and encode the rendered frames.

Refuses to encode if any frame is missing or not a complete PNG.  Encodes H.264 (BT.709, yuv420p, even
dimensions) with the audio track, verifies the result with ffprobe and makes a two-pass copy that stays
just under 25 MB for Instagram DMs.

usage: python tools/encode.py --frames DIR --first 1 --last 1560 --audio a.wav --out film.mp4
       [--ig film_ig.mp4] [--overlay physics.json] [--fps 24] [--report report.json]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_frames import png_complete  # noqa: E402


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def overlay_frames(src, dst, first, last, physics):
    from PIL import Image, ImageDraw, ImageFont
    rows = {r["frame"]: r for r in json.load(open(physics))}
    os.makedirs(dst, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for f in range(first, last + 1):
        im = Image.open(os.path.join(src, f"frame_{f:05d}.png")).convert("RGB")
        r = rows.get(f, {})
        lines = [f"{r.get('shot','')}  frame {f}  t={r.get('t',0):5.2f}s  lens {r.get('lens','')}mm",
                 f"train {r.get('speed_kmh',0):.0f} km/h = {r.get('speed_ms',0):.2f} m/s   s_head {r.get('s_head',0):8.1f} m",
                 f"wheel 920 mm: {r.get('wheel_rps',0):.2f} rev/s = {r.get('wheel_deg_frame',0):.0f} deg/frame   blur 180deg: {r.get('blur_m',0):.2f} m",
                 f"camera {r.get('cam_speed_ms',0):6.2f} m/s   cam-train {r.get('cam_to_train_m',0):7.1f} m   cant {r.get('cant_mm',0):5.1f} mm"]
        d = ImageDraw.Draw(im)
        h = 16 * len(lines) + 8
        d.rectangle([0, im.height - h, im.width, im.height], fill=(0, 0, 0))
        for i, ln in enumerate(lines):
            d.text((6, im.height - h + 4 + 16 * i), ln, fill=(255, 230, 120), font=font)
        im.save(os.path.join(dst, f"frame_{f:05d}.png"), compress_level=1)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True); ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ig", default=None)
    ap.add_argument("--overlay", default=None); ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--report", default=None); ap.add_argument("--maxrate", default="10M")
    a = ap.parse_args()
    missing = [f for f in range(a.first, a.last + 1) if not png_complete(os.path.join(a.frames, f"frame_{f:05d}.png"))]
    if missing:
        print(f"[encode] REFUSING: {len(missing)} frames missing or incomplete, e.g. {missing[:20]}", flush=True)
        sys.exit(2)
    n = a.last - a.first + 1
    print(f"[encode] all {n} frames present and complete", flush=True)
    src = a.frames
    tmp = None
    if a.overlay:
        tmp = tempfile.mkdtemp()
        src = overlay_frames(a.frames, tmp, a.first, a.last, a.overlay)
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709:out_range=tv,format=yuv420p"
    common_in = ["-framerate", str(a.fps), "-start_number", str(a.first), "-i", os.path.join(src, "frame_%05d.png"), "-i", a.audio]
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    run(["ffmpeg", "-y", *common_in, "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-maxrate", a.maxrate, "-bufsize", "24M", "-profile:v", "high", "-level", "4.2", *tags,
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-frames:v", str(n), "-movflags", "+faststart", a.out])
    report = {"main": check(a.out, n, a.fps)}
    if a.ig:
        dur = n / a.fps
        target_bytes = 24.6e6
        v_kbps = int((target_bytes * 8 / dur - 128e3) / 1000 * 0.985)
        log = os.path.join(tempfile.gettempdir(), "x264_2pass")
        base = ["ffmpeg", "-y", *common_in, "-vf", vf, "-c:v", "libx264", "-preset", "veryslow", "-b:v", f"{v_kbps}k",
                "-maxrate", f"{int(v_kbps * 1.6)}k", "-bufsize", f"{int(v_kbps * 2)}k", *tags, "-passlogfile", log, "-frames:v", str(n)]
        run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull])
        run(base + ["-pass", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-movflags", "+faststart", a.ig])
        report["instagram"] = check(a.ig, n, a.fps)
        if report["instagram"]["size_bytes"] >= 25e6:
            print("[encode] Instagram copy exceeds 25 MB", flush=True); sys.exit(4)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(report, indent=1))
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)


def check(path, n, fps):
    info = ffprobe(path)
    v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    au = [s for s in info["streams"] if s["codec_type"] == "audio"]
    frames = int(v.get("nb_read_frames", v.get("nb_frames", 0)))
    dur = float(info["format"]["duration"])
    rep = dict(file=os.path.basename(path), size_bytes=os.path.getsize(path), width=v["width"], height=v["height"],
               codec=v["codec_name"], pix_fmt=v["pix_fmt"], color_space=v.get("color_space"), color_primaries=v.get("color_primaries"),
               color_transfer=v.get("color_transfer"), frames=frames, duration_s=dur, audio=bool(au),
               audio_codec=au[0]["codec_name"] if au else None, bitrate=int(info["format"].get("bit_rate", 0)))
    ok = frames == n and abs(dur - n / fps) < 0.1 and bool(au) and v["width"] % 2 == 0 and v["height"] % 2 == 0
    rep["ok"] = ok
    print(f"[encode] check {path}: {rep}", flush=True)
    if not ok:
        print("[encode] ffprobe verification FAILED", flush=True); sys.exit(5)
    return rep


if __name__ == "__main__":
    main()
