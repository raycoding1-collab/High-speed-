# Handoff: patch shot S9 (frames 1449-1595) of the v3 film

Branch `ccr-ebebc3e5-m6dx41`. Final film: output/cr450_final.mp4 (render run 45256670447, frames kept as
artifacts until ~2026-12-01).

## Two bugs, both in S9 (camera above the tunnel portal, slow-motion ramp, see film/shots.py)
1. Portal plaque "桂岭隧道" (film/build.py `build_tunnel`, material from `infra._sign_material`) is a thin
   single-sided panel: S9 sees it from behind with the text mirrored. Fix: solid back plate / box with a plain
   material so text shows only from the front (check SX 1596-1720 still shows it correctly).
2. Wake dust/sand/leaves (film/wake.py, sampled in sim time) freeze right after the S9 slow-motion ramp ends.
   Find why the particle time stops advancing after the ramp and fix it.

## Test locally first (one heavy Blender process at a time)
`python3 -m film.build --assets assets --out X/scene.blend --quality final --texdir X/textures`, then
`python3 tools/render_still.py X/scene.blend X <frame> --res 960x540 --samples 24` for S9 frames around the ramp end.

## Re-render only S9, then re-encode (GitHub Actions, free; Modal is disabled - do not use it)
1. render.yml: quality=final, jobs=20, frames=1449-1595, overlay=false, encode=false. Re-run failed/short jobs
   (rerun_failed_jobs); jobs resume from checkpoint artifacts.
2. finish.yml: scene_run=<new run>, render_frames="", frame_runs="<new run> 45256670447" (new run FIRST: the
   collector keeps the first copy of each frame), quality=final, encode=true, last=1720, grain=2.5, grade="".
   It commits output/cr450_final*.mp4 + report_final.json and uploads artifact film-final (includes the 4K).
3. Verify report_final.json (1720 frames, 71.67 s, BT.709) and -23 LUFS.
