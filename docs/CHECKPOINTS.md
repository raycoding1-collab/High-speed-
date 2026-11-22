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

## (e) Final film
Workflow `render-film`, quality `final`, 40 jobs (max 20 in parallel), interleaved frames, resumable;
encode job validates all 1560 PNGs, encodes H.264 BT.709 + R128 audio, verifies with ffprobe, commits
`output/cr450_final.mp4` and `output/cr450_final_instagram.mp4` (< 25 MB, two-pass).
