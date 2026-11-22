# v3 verification matrix (gate checks before the full render)

Evidence from cloud test run [45074102764](https://github.com/raycoding1-collab/High-speed-/actions/runs/45074102764)
(full scene build, 1920x1080, 48 spp, 10 frames covering all 9 shots; reports in `output/tests/45074102764/`).

| Gate | Status | Method | Measured evidence |
|---|---|---|---|
| Every instanced asset renders (no NaN transforms) | PASS | `tools/verify_scene.py`: evaluated depsgraph at the middle frame of every shot | 470,130 instances per shot, 0 invalid (`verify_scene.json`) |
| All textures available on the render machines | PASS | same: every FILE image packed or on disk | 124 / 124 packed, 0 missing |
| Worldbuilding present | PASS | object counts at build + visual check in S1, S2, S4, S6 frames | 190 houses in 17 hamlets (7 placed in camera views), 157 power poles, 37 fish ponds, 395 straw stacks, 6 greenhouse sites, 1,145 orchard trees, 26 tombs, ~21,000 grove trees, ~1,300 bamboo clumps, ~158,000 trees on the hill massif |
| Exposure / clipping | PASS | `tools/verify_frames.py` on the full-res frames | clipped <= 0.12 %, crushed <= 1.87 % (S7, deep cutting), mean luma 0.30-0.57 |
| Shadow colour neutrality | PASS | mean (G+B)/2 - R of the darkest 25 % of pixels | -13.1 ... +7.1 (limit +8; v1 teal-lift bug was ~+14) |
| Loudness | PASS | pyloudnorm BS.1770 integrated + 4x oversampled true peak | -23.00 LUFS, -5.07 dBTP, 63.50 s (`verify_audio.json`) |
| Film length / edit | PASS | shot timeline | 1524 frames = 63.5 s @ 24 fps; S9 cut 2.0 s after the tail enters the tunnel |
| Render cost | measured | Modal L40S per-frame times (`timing_modal.json`) | 63-104 s/frame on GPU (v2: 44 s) -> ~7-8 min/frame on a 4-core runner -> 60 jobs in 3 batches of 20, ~3.2 h per job (deadline 5.6 h), ~9-10 h wall |
| Motion (camera paths, jolt, timing) | running | low-res motion preview of the whole film with physics overlay (render.yml, preview) | pending |

Changes after this test run (re-tested locally before the full render): S4 helicopter now flies ahead of the
train and looks back with the sun behind the lens (front-lit nose); S2 tilted down 6 m of aim height with three
real trees framing the field; both verified in local test frames.
