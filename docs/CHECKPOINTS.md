# Checkpoints (a)–(e)

The user authorised finishing the whole production while away ("make sure everything works exactly well
and finish it fully"). Every checkpoint below was still run and checked before the next step.

## (a) Train vs real reference photos
- `checkpoints/a1_train_side_render.jpg`: rendered from the **solved camera** of Commons photo
  *CR450AF-0201 at Beijingdong Railway Station* (OpenCV calibration, 5.8 px RMS). Silhouette, nose profile,
  door/window/stripe positions, gangway and skirt line coincide with the photo.
- `checkpoints/a2_train_front_render.jpg`: front three-quarter view comparable to Commons photo
  *CR450AF-0201 at CARS Testing Track (20250213152718)*: black canopy with gold emblem, bill, deflector shelf.
- Side-by-side sheets (with the CC BY-SA photos) were sent privately and are not committed to this public repo.
- Known deviation: the real nose is slightly fuller in plan than the model.

## (b) Environment stills (1080p, final quality)
Rendered in the cloud timing run 44975074671 (artifacts `frames-0..8-a1`, one frame per shot).

## (c) Low-resolution motion preview with physics numbers
Workflow run 44975008213 (640×360, 16 spp): every frame carries shot, time, speed (400 km/h = 111.11 m/s),
wheel rate (38.44 rev/s = 576.6°/frame), motion blur length (2.31 m), camera speed and distance to the train,
cant. Output: `output/cr450_preview.mp4` (+ `_instagram.mp4`, ffprobe report).

## (d) Render-time estimate (measured on GitHub ubuntu-24.04 4-core runners)
| Shot | s/frame @1920×1080, 48 spp (incl. ~40 s scene load) |
|---|---|
| S1 aerial | 210 |
| S2 paddy | 293 |
| S3 drone | 252 |
| S4 helicopter | 350 |
| S5 cab | 498 |
| S6 pantograph | 382 |
| S7 cutting | 508 |
| S8 600 mm | 304 |
| S9 portal | 293 |
| **mean** | **~340 s (5.7 min)** |

1560 frames × 5.7 min ≈ 148 runner-hours.
- 20 jobs: ~7.4 h per job → exceeds the 6 h limit.
- **40 jobs (two batches of 20): ~3.7 h per job**, total ≈ 7.5–8 h wall clock → chosen.
- 4K would need ≈ 590 runner-hours (≈ 6 batches, ~33 h) → not within a sensible budget; 1080p chosen.

## (e) Final film (delivered 2026-11-22 13:19 UTC)
- Main render: run [44982706751](https://github.com/raycoding1-collab/High-speed-/actions/runs/44982706751), 40 jobs × 39 interleaved
  frames, 1920×1080, 48 spp. Measured: 164–276 min per job (mean ≈ 206 min ≈ 310 s/frame incl. scene load),
  ≈ 140 runner-hours. Job 34 lost its runner after 1 h 50 min (no log, no artifact) and job 35 ran on a slow runner.
- Recovery without re-running whole jobs: workflow `finish.yml` re-rendered only those 2 × 39 frames with the
  main run's own packed scene (runs [45013365198](https://github.com/raycoding1-collab/High-speed-/actions/runs/45013365198) and
  [45023138633](https://github.com/raycoding1-collab/High-speed-/actions/runs/45023138633), 19–20 jobs × 2 frames, ~12 min each).
- Encode: run [45025171197](https://github.com/raycoding1-collab/High-speed-/actions/runs/45025171197) collected all 1560 frames from the
  three runs, refused nothing (all present and complete), encoded and committed:

| File | Size | Video | Frames / duration | Audio |
|---|---|---|---|---|
| `output/cr450_final.mp4` (master) | 40.8 MB | 1920×1080 H.264 High, BT.709, 5.0 Mbit/s | 1560 / 65.0 s | AAC 256k, −23.0 LUFS, TP −5.6 dB |
| `output/cr450_final_instagram.mp4` (DM) | 24.2 MB | 1920×1080 H.264 two-pass veryslow, 3.0 Mbit/s | 1560 / 65.0 s | AAC 128k |
| `cr450_final_youtube_4k.mp4` (artifact `film-final` only) | 147.3 MB | 3840×2160 lanczos upscale, H.264 18 Mbit/s | 1560 / 65.0 s | AAC 320k |

- Loudness measured on the master with ffmpeg ebur128: I −23.0 LUFS, LRA 14.8 LU, true peak −5.6 dBFS.

### Honest quality review of the final frames
Strong: train shape/livery and its motion (true 400 km/h, blur, wheel rate), OCS masts and wires, viaduct, track,
golden-hour sky, the low cutting pass-by with flying leaves, the sound.
Below the photoreal bar (reads as high-quality CGI, not real footage): the valley floor (flat fields with stripe
shadows and some flat grey/white patches), smooth "blob" karst hills, and the dark cutting/portal slopes with
bright lattice lines. Fixing these needs scene changes and another full render.

## v2 (2026-11-22): fixes after the v1 review, hybrid render
- Fixed: every scattered instance was NaN (geometry-nodes wake 0 x inf) -> all trees/rice/crowns invisible in v1;
  valley tree budget spent outside the region; teal shadows from a compositor lift on linear data; S7/S9 exposure.
- Render: Modal 10 x L40S for S4-S7 (648 frames, 48 min, 43.7 s/frame, run 45043063354) + GitHub 19 jobs for
  S1-S3, S8-S9 (912 frames, run 45043444866; 6 jobs hit the 5.6 h deadline a few frames short and were re-run
  with resume) -> merged and encoded with finish.yml.

## v3 (in progress): realism + worldbuilding pass (from the user's improvement list and the Gemini comparison)
- New: Guangxi house kit (5 types, real openings, balconies, tiled/pantile roofs, solar heaters, walled yards),
  hamlets placed where the cameras look, village roads with ditches, power lines, fish ponds, straw stacks,
  greenhouses, citrus orchards, hillside tombs, bamboo on the river banks, cauliflower-crown groves, forested hill
  massif with karst knobs; real 3D lattice beams, concrete side drains and gravel cess in the cutting; detailed cab
  console with DMI displays; concrete rain streaks / per-span tone / pier-foot soiling; polished anisotropic rail
  heads; pressure-wave camera jolt at the trackside pass; S4 helicopter re-shot front-lit; S9 cut 2 s after the
  tail; singing rails, pressure thump, traction hum and mast whooshes in the sound; lens dispersion.
- Gate evidence: docs/V3_VERIFICATION.md.
